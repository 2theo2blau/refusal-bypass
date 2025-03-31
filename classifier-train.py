import torch
import numpy as np
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configuration
# Dataset
DATASET_NAME = "allenai/wildguardmix"
TEXT_FIELD = "response"
LABEL_FIELD = "response_refusal_label"
TRAIN_CONFIG_NAME = "wildguardtrain"
TEST_CONFIG_NAME = "wildguardtest"

# Model
MODEL = "roberta-large"

# Training
OUTPUT_DIR = "refusal-classifier"

LEARNING_RATE = 1e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
WEIGHT_DECAY = 0

FP16 = torch.cuda.is_available() # use mixed precision if available

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

# Load dataset
print(f"Loading dataset {DATASET_NAME}...")
try:
    # Load train and test splits separately using their respective configs
    raw_train_dataset = load_dataset(DATASET_NAME, TRAIN_CONFIG_NAME, split='train')
    raw_eval_dataset = load_dataset(DATASET_NAME, TEST_CONFIG_NAME, split='test')
    # Combine into a DatasetDict for easier handling if needed later, though we'll process separately
    # raw_datasets = DatasetDict({"train": raw_train_dataset, "test": raw_eval_dataset})
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Preprocessing
print("Starting preprocessing...")

label_map = {
    "refusal": 1,
    "compliance": 0
}
num_labels = len(label_map)

def preprocess_function(examples):
    texts = [str(text) if text is not None else "" for text in examples[TEXT_FIELD]]
    
    labels = []
    for label_str in examples[LABEL_FIELD]:
        labels.append(label_map.get(label_str, -1))
    return {"text": texts, "label_int": labels}


required_columns = [TEXT_FIELD, LABEL_FIELD]
print(f"Filtering and mapping columns {TEXT_FIELD} and {LABEL_FIELD}...")

# Process train dataset
print("Processing train dataset...")
processed_train_dataset = raw_train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=[col for col in raw_train_dataset.column_names if col not in required_columns]
)
processed_train_dataset = processed_train_dataset.filter(lambda example: example['label_int'] != -1)
processed_train_dataset = processed_train_dataset.rename_column("label_int", "label")

# Process eval dataset
print("Processing eval dataset...")
processed_eval_dataset = raw_eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=[col for col in raw_eval_dataset.column_names if col not in required_columns]
)
processed_eval_dataset = processed_eval_dataset.filter(lambda example: example['label_int'] != -1)
processed_eval_dataset = processed_eval_dataset.rename_column("label_int", "label")


# Tokenization
print(f"Loading tokenizer for {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

print("Tokenizing dataset...")

# Tokenize train dataset
print("Tokenizing train dataset...")
tokenized_train_dataset = processed_train_dataset.map(tokenize_function, batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
tokenized_train_dataset.set_format("torch")

# Tokenize eval dataset
print("Tokenizing eval dataset...")
tokenized_eval_dataset = processed_eval_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["text"])
tokenized_eval_dataset.set_format("torch")


print("Dataset tokenization complete.")
print("Example tokenized train entry:")
print(tokenized_train_dataset[0].keys())

train_dataset = tokenized_train_dataset
eval_dataset = tokenized_eval_dataset

# Model Loading
print(f"Loading model {MODEL}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=num_labels
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", pos_label=1, zero_division=0)
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else None,
    fp16=FP16,
    seed=42,
    logging_steps=20,
    save_total_limit=2,

)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
)

# Training
print("Starting training...")
try:
    train_result = trainer.train()
    print("Training complete.")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Evaluation
print("Evaluating model on test split...")
try:
    eval_metrics = trainer.evaluate()
    print("Validation split metrics:")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit(1)

print("Training and evaluation complete.")

# Save final model
print("Saving best checkpoint...")
try:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)

print("Training complete. Exiting.")



