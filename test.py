import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

# Load the saved model and tokenizer
model_path = "distilbert-finetuned-mrpc"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set up the pipeline for text classification
classifier = pipeline("text-classification", model=model_path)

# Function to predict sentiment for a pair of sentences
def predict(text1, text2):
    inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# Example usage
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A fast, dark-colored fox leaps over a sleepy dog."
print("Predicted class:", predict(text1, text2))  # Outputs prediction

# Alternatively, using the classifier pipeline
text = "The movie was fantastic!"
result = classifier(text)
print("Pipeline prediction:", result)