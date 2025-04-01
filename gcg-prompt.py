import argparse
import random
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class GCGSuffixOptimizer:
    def __init__(self,
                 model_name,
                 target_prefix="Sure, here is",
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 suffix_len=20,
                 batch_size=512,
                 top_k=256,
                 num_steps=500):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            print("Warning: pad_token is not set, using eos_token as pad_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.target_prefix = target_prefix
        self.suffix_len = suffix_len
        self.batch_size = batch_size
        self.top_k = top_k
        self.num_steps = num_steps

        self.top_k = min(top_k, self.tokenizer.vocab_size)

        self.device = device

        # Get embedding matrix, checking common locations
        if hasattr(self.model, 'get_input_embeddings'):
            self.embedding_matrix = self.model.get_input_embeddings().weight.data
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            self.embedding_matrix = self.model.model.embed_tokens.weight.data
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            self.embedding_matrix = self.model.transformer.wte.weight.data
        else:
            raise ValueError("Could not find embedding matrix in the model")
        
    def tokenize_and_prepare(self, prompt_str):
        # Fix tokenization to properly handle the output
        prompt_encoding = self.tokenizer(prompt_str, return_tensors='pt', add_special_tokens=False)
        prompt_tokens = prompt_encoding['input_ids'][0]
        target_encoding = self.tokenizer(self.target_prefix, return_tensors='pt', add_special_tokens=False)
        target_tokens = target_encoding['input_ids'][0]

        prompt_len = len(prompt_tokens)
        target_len = len(target_tokens)

        suffix_indices = torch.arange(prompt_len, prompt_len + self.suffix_len)
        # target_indices = torch.arange(prompt_len, self.suffix_len, prompt_len + self.suffix_len + target_len)
        target_start_idx = prompt_len + self.suffix_len
        target_end_idx = target_start_idx + target_len
        target_indices = torch.arange(target_start_idx, target_end_idx)

        return prompt_tokens.to(self.device), target_tokens.to(self.device), \
            suffix_indices.to(self.device), target_indices.to(self.device)
    
    def calculate_loss(self, input_ids_batch, target_indices, target_tokens):
        batch_size = input_ids_batch.shape[0]

        target_indices = target_indices.to(self.device)
        target_tokens = target_tokens.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_batch)
            logits_batch = outputs.logits

            target_indices_adjusted = target_indices - 1
            target_logits = logits_batch[:, target_indices_adjusted, :]

            target_logits_flat = target_logits.view(-1, self.tokenizer.vocab_size)
            target_tokens_flat = target_tokens.repeat(batch_size)
            # cross-entropy loss
            per_token_loss = F.cross_entropy(target_logits_flat, target_tokens_flat, reduction='none')
            per_token_loss = per_token_loss.view(batch_size, -1)
            loss_per_batch_item = per_token_loss.mean(dim=1)
            
            # cleanup
            del full_input_ids, outputs, logits, target_logits
            torch.cuda.empty_cache()

            return loss_per_batch_item
        
    def calculate_gradient_and_candidates(self, input_ids, suffix_indices, target_indices, target_tokens):
        # Create input embeddings with requires_grad=True from the start
        input_len = len(input_ids)
        target_len = len(target_tokens)
        full_len = input_len + target_len

        full_input_ids = torch.cat([input_ids, target_tokens], dim=0).to(self.device)
        full_embeds = self.embedding_matrix[full_input_ids]

        input_embeds = full_embeds[:input_len].clone()
        target_embeds = full_embeds[input_len:].clone()
        input_embeds.requires_grad = True

        final_embeds = torch.cat([input_embeds, target_embeds], dim=0).unsqueeze(0)

        # forward pass w/ embeddings
        outputs = self.model(inputs_embeds=final_embeds)
        logits = outputs.logits[0]

        # calculate loss for target prefix
        target_loss_slice_start = input_len - 1
        target_loss_slice_end = full_len - 1
        target_logits = logits[target_loss_slice_start:target_loss_slice_end, :]

        if target_logits.shape[0] != target_len:
            raise ValueError("Target logits shape does not match target tokens length")

        loss = F.cross_entropy(target_logits, target_tokens)

        # backward pass
        self.model.zero_grad()
        loss.backward()

        # gradient of loss w.r.t. input embeddings
        embed_gradients = input_embeds.grad

        # select gradients for only suffix positions
        suffix_gradients = embed_gradients[suffix_indices, :]

        # approximate gradient w.r.t. one-hot vectors
        token_gradient_scores = suffix_gradients @ self.embedding_matrix.T.to(self.device)

        # get top k candidates
        _, top_k_indices = torch.topk(-token_gradient_scores, self.top_k, dim=1)

        top_k_indices = top_k_indices.detach()
        del full_input_ids, full_embeds, input_embeds, target_embeds, final_embeds
        del outputs, logits, target_logits, loss, embed_gradients, suffix_gradients, token_gradient_scores
        torch.cuda.empty_cache()

        return top_k_indices
    
    def optimize_suffix(self, prompt_str):
        print("Starting GCG suffix optimization...")
        start_time = time.time()

        prompt_tokens, target_tokens, suffix_indices, target_indices = self.tokenize_and_prepare(prompt_str)
        prompt_len = len(prompt_tokens)

        current_suffix_tokens = torch.randint(0, self.tokenizer.vocab_size, (self.suffix_len,), device=self.device)
        
        best_suffix_tokens = current_suffix_tokens.clone()
        best_loss = float('inf')

        print(f"Initial loss: {best_loss:.4f}")

        for step in range(self.num_steps):
            step_start_time = time.time()

            current_input_ids = torch.cat([prompt_tokens, current_suffix_tokens], dim=0)

            # candidate generation
            try:
                top_k_candidates_per_pos = self.calculate_gradient_and_candidates(current_input_ids, suffix_indices, target_indices, target_tokens)
            except Exception as e:
                print(f"Error calculating gradient at step {step + 1}: {e}")
                print("Skipping gradient calculation for this step")
                continue

            # # batch evaluation
            # candidate_losses = []
            # candidate_info = []

            # positions_to_try = random.choices(range(self.suffix_len), k=self.batch_size)
            # replacement_tokens = []
            # for pos_idx in positions_to_try:
            #     candidate_token_idx = random.choice(range(self.top_k))
            #     replacement_tokens.append(top_k_candidates_per_pos[pos_idx, candidate_token_idx].item())
            
            # temp_suffix_tokens_batch = current_suffix_tokens.unsqueeze(0).repeat(self.batch_size, 1)

            # row_indices = torch.arange(self.batch_size)
            # col_indices = torch.tensor(positions_to_try)
            # temp_suffix_tokens_batch[row_indices, col_indices] = torch.tensor(replacement_tokens, device=self.device)

            # batch_input_ids = torch.cat([
            #     prompt_tokens.unsqueeze(0).repeat(self.batch_size, 1),
            #     temp_suffix_tokens_batch
            # ], dim=1)

            # current_batch_losses = []
            # for i in range(self.batch_size):
            #     loss = self.calculate_loss(batch_input_ids[i], target_indices, target_tokens)
            #     current_batch_losses.append(loss)

            #     candidate_info.append({
            #         'pos': positions_to_try[i],
            #         'token_id': replacement_tokens[i],
            #         'loss': loss
            #     })

            # if not candidate_info:
            #     print(f"No candidates saved at step {step + 1}.")
            #     continue

            positions_to_try = random.choices(range(self.suffix_len), k=self.batch_size)
            replacement_tokens = []
            for pos_idx in positions_to_try:
                if pos_idx < top_k_candidates_per_pos.shape[0]:
                    candidate_token_idx = random.choice(range(self.top_k))
                    replacement_tokens.append(top_k_candidates_per_pos[pos_idx, candidate_token_idx].item())
                else:
                    print(f"Warning: pos_idx {pos_idx} is out of bounds for top_k_candidates_per_pos. Shape: {top_k_candidates_per_pos.shape}")
                    replacement_tokens.append(current_suffix_tokens[pos_idx].item())

            temp_suffix_tokens_batch = current_suffix_tokens.unsqueeze(0).repeat(self.batch_size, 1)

            row_indices = torch.arange(self.batch_size, device=self.device)
            col_indices = torch.tensor(positions_to_try, device=self.device)
            replacement_tokens_tensor = torch.tensor(replacement_tokens, device=self.device)

            valid_indices_mask = (col_indices >= 0) & (col_indices < self.suffix_len)
            if not torch.all(valid_indices_mask):
                print(f"Warning: Invalid column indices detected during batch update. Clamping.")
                col_indices = torch.clamp(col_indices, 0, self.suffix_len - 1)

            temp_suffix_tokens_batch.scatter_(1, col_indices.unsqueeze(1), replacement_tokens_tensor.unsqueeze(1))
            
            batch_input_ids = torch.cat([
                prompt_tokens.unsqueeze(0).repeat(self.batch_size, 1),
                temp_suffix_tokens_batch
            ], dim=1)

            batch_losses = self.calculate_loss(batch_input_ids, target_indices, target_tokens)

            if batch_losses.numel() == 0:
                print(f"Warning: No valid losses calculated for batch. Skipping this step.")
                continue

            best_candidate_idx_in_batch = torch.argmin(batch_losses)
            min_batch_loss = batch_losses.min().item()

            # update current suffix with best candidate
            best_pos = positions_to_try[best_candidate_idx_in_batch]
            best_token_id = replacement_tokens[best_candidate_idx_in_batch]
            current_suffix_tokens = current_suffix_tokens.clone()
            current_suffix_tokens[best_pos] = best_token_id

            # Track best suffix found so far
            if min_batch_loss < best_loss:
                best_loss = min_batch_loss
                best_suffix_tokens = current_suffix_tokens.clone()
                print(f"Step {step + 1}/{self.num_steps} | New Best Loss: {best_loss:.4f} **")
            else:
                print(f"Step {step + 1}/{self.num_steps} | Current Loss: {min_batch_loss:.4f} (Best: {best_loss:.4f})", end='\r')

            step_time = time.time() - step_start_time

            if (step + 1) % 20 == 0:
                print(f"\nSuffix @ step {step + 1}: {self.tokenizer.decode(best_suffix_tokens)}")

        total_time = time.time() - start_time
        print(f"\nOptimization complete! Total time: {total_time:.2f}s")
        print(f"Final suffix: {self.tokenizer.decode(best_suffix_tokens)}")
        print(f"Final loss: {best_loss:.4f}")

        optimized_suffix = self.tokenizer.decode(best_suffix_tokens, skip_special_tokens=True)

        return optimized_suffix
    

    def generate_with_suffix(self, prompt_str, suffix_str, max_new_tokens=100):
        full_prompt = prompt_str + suffix_str

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # temperature = 0.7, top_p = 0.9, do_sample=True
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCG Suffix Optimizer")
    parser.add_argument("--model", type=str, required=True, help="huggingface model name")
    parser.add_argument("--prompt", type=str, required=True, help="original user prompt")
    parser.add_argument("--target_prefix", type=str, default="Sure, here is", required=True, help="desired target output prefix")
    parser.add_argument("--suffix_len", type=int, default=20, help="length of suffix to optimize")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size for optimization")
    parser.add_argument("--top_k", type=int, default=256, help="number of top candidates to consider")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="max new tokens")
    parser.add_argument("--num_steps", type=int, default=500, help="number of optimization steps")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="device to run on")

    args = parser.parse_args()

    try:
        optimizer = GCGSuffixOptimizer(
            model_name=args.model,
            target_prefix=args.target_prefix,
            device=args.device,
            suffix_len=args.suffix_len,
            batch_size=args.batch_size,
            top_k=args.top_k,
            num_steps=args.num_steps
        )

        optimized_suffix = optimizer.optimize_suffix(args.prompt)
        
        print("\n--- Optimization Complete ---")
        print(f"Original Prompt : '{args.prompt}'")
        print(f"Target Prefix   : '{args.target_prefix}'")
        print(f"Optimized Suffix: '{optimized_suffix}'")
        print("----------------------------")

        # Demonstrate generation with the optimized suffix
        print("\nAttempting generation with optimized suffix...")
        generated_output = optimizer.generate_with_suffix(args.prompt, optimized_suffix)
        print("\n--- Generated Output ---")
        print(generated_output)
        print("------------------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    
    