
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Connect to the NewsArticles.db and extract headlines
conn = sqlite3.connect("NewsArticles.db")
cursor = conn.cursor()
cursor.execute("SELECT headline FROM articles WHERE headline IS NOT NULL")
financial_texts = [row[0] for row in cursor.fetchall()]
conn.close()

inputs = tokenizer(
    financial_texts,
    padding=True,
    truncation=True,
    return_tensors='pt'  # Return PyTorch tensors
)

outputs = model(**inputs)

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# You can then get the predicted sentiment label
predicted_labels = torch.argmax(probabilities, dim=-1)

# Map the numerical labels to sentiment words (e.g., 0: 'positive', 1: 'negative', 2: 'neutral')
sentiment_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'} # Adjust based on model's output order

for i, text in enumerate(financial_texts):
    sentiment = sentiment_mapping[predicted_labels[i].item()]
    print(f"Text: \"{text}\" -> Predicted Sentiment: {sentiment}")