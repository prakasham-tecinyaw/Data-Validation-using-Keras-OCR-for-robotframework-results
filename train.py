import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset (using MRPC as an example)
dataset = load_dataset("glue", "mrpc")

# Load the tokenizer and model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",                # Output directory
    evaluation_strategy="epoch",           # Evaluation strategy to adopt during training
    learning_rate=2e-5,                    # Learning rate
    per_device_train_batch_size=16,        # Batch size for training
    per_device_eval_batch_size=16,         # Batch size for evaluation
    num_train_epochs=3,                     # Total number of training epochs
    weight_decay=0.01,                     # Strength of weight decay
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                            # The instantiated 🤗 Transformers model to be trained
    args=training_args,                    # Training arguments, defined above
    train_dataset=tokenized_datasets["train"],  # Training dataset
    eval_dataset=tokenized_datasets["validation"],  # Evaluation dataset
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("distilbert-finetuned-mrpc")
tokenizer.save_pretrained("distilbert-finetuned-mrpc")

# Example of making predictions using the pipeline
from transformers import pipeline

# Use the saved model for predictions
classifier = pipeline("text-classification", model="distilbert-finetuned-mrpc")

# Example usage
def predict(text1, text2):
    inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# Example usage
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A fast, dark-colored fox leaps over a sleepy dog."
print(predict(text1, text2))  # Outputs prediction