import argparse
import random
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from datetime import datetime

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
        # # The old target_indices assumed target followed suffix directly in a longer hypothetical sequence
        # target_start_idx = prompt_len + self.suffix_len
        # target_end_idx = target_start_idx + target_len
        # target_indices = torch.arange(target_start_idx, target_end_idx)

        return prompt_tokens.to(self.device), target_tokens.to(self.device), \
            suffix_indices.to(self.device), target_len # Return target_len instead of target_indices
    
    # Modify calculate_loss to accept the full sequence batch and target length
    def calculate_loss(self, full_input_ids_batch, target_len, target_tokens):
        # full_input_ids_batch has shape (batch_size, prompt_len + suffix_len + target_len)
        batch_size = full_input_ids_batch.shape[0]
        seq_len = full_input_ids_batch.shape[1] # P + L + T
        prompt_suffix_len = seq_len - target_len # P + L

        # Ensure target_tokens is on the correct device and expanded for the batch
        target_tokens = target_tokens.to(self.device) # Shape (T,)
        target_tokens_batch = target_tokens.unsqueeze(0).repeat(batch_size, 1) # Shape (B, T)
        
        with torch.no_grad():
            outputs = self.model(input_ids=full_input_ids_batch)
            logits_batch = outputs.logits # Shape (B, P + L + T, V)
            
            # Get the actual vocabulary size from the logits, not the tokenizer
            vocab_size = logits_batch.shape[-1]
            tokenizer_vocab_size = self.tokenizer.vocab_size
            
            if vocab_size != tokenizer_vocab_size:
                print(f"Note: Model's vocabulary size ({vocab_size}) differs from tokenizer's reported size ({tokenizer_vocab_size})")
            
            # Safety check for sequence length
            if logits_batch.shape[1] < prompt_suffix_len:
                print(f"Warning: Logits sequence length ({logits_batch.shape[1]}) is shorter than expected prompt+suffix length ({prompt_suffix_len})")
                # Adjust prompt_suffix_len to avoid index errors
                prompt_suffix_len = logits_batch.shape[1]
            
            # Indices for logits that predict the target tokens - ensure we don't go out of bounds
            start_idx = max(0, prompt_suffix_len - 1)
            end_idx = min(logits_batch.shape[1] - 1, start_idx + target_len)
            actual_target_len = end_idx - start_idx
            
            if actual_target_len < target_len:
                print(f"Warning: Could only get {actual_target_len} target positions out of {target_len} expected")
            
            target_logit_indices = torch.arange(start_idx, end_idx, device=self.device)
            
            # Ensure we're not trying to access indices beyond what's available
            if target_logit_indices.numel() == 0:
                print("Error: No valid target logit indices available")
                # Return a default high loss value to avoid stopping the optimization
                return torch.ones(batch_size, device=self.device) * 100.0

            # Select the relevant logits
            target_logits = logits_batch[:, target_logit_indices, :]
            
            try:
                # Use actual dimensions for reshaping
                batch_dim, seq_dim, vocab_dim = target_logits.shape
                target_logits_flat = target_logits.reshape(batch_dim * seq_dim, vocab_dim)
                
                # Adjust target_tokens_batch to match the actual sequence dimension
                if target_tokens_batch.shape[0] != batch_dim or target_tokens_batch.shape[1] != seq_dim:
                    # Trim or pad target_tokens_batch to match seq_dim
                    if target_tokens_batch.shape[1] > seq_dim:
                        target_tokens_batch = target_tokens_batch[:batch_dim, :seq_dim]
                    elif target_tokens_batch.shape[1] < seq_dim:
                        padding = torch.zeros(batch_dim, seq_dim - target_tokens_batch.shape[1], 
                                              device=self.device, dtype=torch.long)
                        target_tokens_batch = torch.cat([target_tokens_batch, padding], dim=1)
                
                target_tokens_flat = target_tokens_batch.reshape(-1)
                
                # cross-entropy loss - ensure tokens are within the vocabulary range
                target_tokens_flat = torch.clamp(target_tokens_flat, 0, vocab_dim - 1)
                per_token_loss = F.cross_entropy(target_logits_flat, target_tokens_flat, reduction='none')
                
                # Reshape per_token_loss back to batch dimensions
                per_token_loss = per_token_loss.view(batch_dim, seq_dim)
                
                # Calculate mean loss per batch item
                loss_per_batch_item = per_token_loss.mean(dim=1)
                
            except Exception as e:
                print(f"Error during reshape/loss calculation: {e}")
                print(f"Target logits shape: {target_logits.shape}, Model vocab size: {vocab_size}")
                print(f"Target tokens batch shape: {target_tokens_batch.shape}")
                print(f"Target tokens min: {target_tokens.min().item()}, max: {target_tokens.max().item()}")
                
                # Return a default loss to avoid stopping optimization
                return torch.ones(batch_size, device=self.device) * 100.0
            
            # cleanup
            del outputs, logits_batch, target_logits, target_logits_flat, target_tokens_flat
            del per_token_loss, target_tokens_batch
            torch.cuda.empty_cache()

            return loss_per_batch_item
        
    def calculate_gradient_and_candidates(self, input_ids, suffix_indices, target_tokens):
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
        
        # Create a timestamped filename for this optimization run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_short = self.model.config._name_or_path.split('/')[-1] if hasattr(self.model, 'config') else "model"
        json_filename = f"gcg_optimization_{model_name_short}_{timestamp}.json"
        
        # Initialize the JSON data structure
        optimization_data = {
            "model": self.model.config._name_or_path if hasattr(self.model, 'config') else "unknown",
            "prompt": prompt_str,
            "target_prefix": self.target_prefix,
            "suffix_len": self.suffix_len,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "num_steps": self.num_steps,
            "start_time": timestamp,
            "suffixes": []
        }

        # Get target_len instead of target_indices
        prompt_tokens, target_tokens, suffix_indices, target_len = self.tokenize_and_prepare(prompt_str)
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
                # Pass only necessary arguments: input_ids (P+L), suffix_indices, target_tokens
                top_k_candidates_per_pos = self.calculate_gradient_and_candidates(current_input_ids, suffix_indices, target_tokens)
            except Exception as e:
                print(f"Error calculating gradient at step {step + 1}: {e}")
                print("Skipping gradient calculation for this step")
                continue

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
            
            # Create the full input batch including target tokens
            prompt_batch = prompt_tokens.unsqueeze(0).repeat(self.batch_size, 1) # Shape (B, P)
            target_batch = target_tokens.unsqueeze(0).repeat(self.batch_size, 1) # Shape (B, T)
            
            # Concatenate: Prompt + Suffix Candidates + Target
            full_batch_input_ids = torch.cat([
                prompt_batch,             # (B, P)
                temp_suffix_tokens_batch, # (B, L)
                target_batch              # (B, T)
            ], dim=1) # Shape (B, P+L+T)


            # Pass the full batch, target_len, and original target_tokens to calculate_loss
            batch_losses = self.calculate_loss(full_batch_input_ids, target_len, target_tokens)

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

            # Calculate step time
            step_time = time.time() - step_start_time
            
            # Track best suffix found so far
            if min_batch_loss < best_loss:
                best_loss = min_batch_loss
                best_suffix_tokens = current_suffix_tokens.clone()
                suffix_str = self.tokenizer.decode(best_suffix_tokens)
                print(f"Step {step + 1}/{self.num_steps} | New Best Loss: {best_loss:.4f} **")
                
                # Add to JSON data every time we find a new best suffix
                optimization_data["suffixes"].append({
                    "step": step + 1,
                    "loss": best_loss,
                    "suffix": suffix_str,
                    "type": "best",
                    "time": time.time() - start_time
                })
                
                # Write the updated data to the JSON file
                with open(json_filename, 'w') as f:
                    json.dump(optimization_data, f, indent=2)
            else:
                print(f"Step {step + 1}/{self.num_steps} | Current Loss: {min_batch_loss:.4f} (Best: {best_loss:.4f})", end='\r')

            # Every 20 steps, log the current suffix
            if (step + 1) % 20 == 0:
                suffix_str = self.tokenizer.decode(best_suffix_tokens)
                print(f"\nSuffix @ step {step + 1}: {suffix_str}")
                
                # Add to JSON data at the periodic checkpoints
                optimization_data["suffixes"].append({
                    "step": step + 1,
                    "loss": best_loss,
                    "suffix": suffix_str,
                    "type": "periodic",
                    "time": time.time() - start_time
                })
                
                # Write the updated data to the JSON file
                with open(json_filename, 'w') as f:
                    json.dump(optimization_data, f, indent=2)

        # End of optimization
        total_time = time.time() - start_time
        print(f"\nOptimization complete! Total time: {total_time:.2f}s")
        
        # Get the final suffix
        final_suffix = self.tokenizer.decode(best_suffix_tokens, skip_special_tokens=True)
        print(f"Final suffix: {final_suffix}")
        print(f"Final loss: {best_loss:.4f}")
        
        # Add final entry to JSON data
        optimization_data["suffixes"].append({
            "step": self.num_steps,
            "loss": best_loss,
            "suffix": final_suffix,
            "type": "final",
            "time": total_time
        })
        optimization_data["total_time"] = total_time
        optimization_data["final_loss"] = best_loss
        
        # Write the final data to the JSON file
        with open(json_filename, 'w') as f:
            json.dump(optimization_data, f, indent=2)
            
        print(f"Optimization data saved to {json_filename}")

        return final_suffix
    

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

    
    