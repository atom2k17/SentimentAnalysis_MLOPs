# sentiment_server.py
from flask import Flask, render_template
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import mlflow
import mlflow.pyfunc

app = Flask(__name__)

# Start MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CryptoSentimentAnalysis")  # Set experiment by name

# Crypto websites to extract sentiment
CRYPTO_SITES = [
    "https://www.coindesk.com",
    "https://cointelegraph.com",
    "https://www.cryptopolitan.com",
    "https://decrypt.co",
    "https://www.cryptonewsz.com"
]

# Load FinBERT (TensorFlow version)
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Define a PyFunc wrapper for FinBERT
class FinBertWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = TFAutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()
        results = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="tf", truncation=True, max_length=512, padding=True)
            outputs = self.model(inputs)
            probs = tf.nn.softmax(outputs.logits, axis=-1)[0].numpy()
            sentiment = ["neutral", "positive", "negative"][np.argmax(probs)]
            results.append(sentiment)
        return results

# Log model with MLflow (only once or conditionally)
with mlflow.start_run(run_name="FinBERT-Sentiment-Analysis"):
    mlflow.pyfunc.log_model(
        artifact_path="finbert_model",
        python_model=FinBertWrapper(),
        registered_model_name="finbert_model"
    )
    mlflow.log_param("model_type", "PyFunc-HF-TF")
    mlflow.set_tag("framework", "TensorFlow")
    mlflow.set_tag("domain", "Crypto News Sentiment")

def analyze_sentiment(texts):
    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="tf", truncation=True, max_length=512, padding=True)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)[0].numpy()
        sentiment = ["neutral", "positive", "negative"][np.argmax(probs)]
        sentiments.append(sentiment)
    return sentiments

def fetch_crypto_prices():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,ripple&vs_currencies=inr,usd"
    try:
        response = requests.get(url)
        data = response.json()
        return {
            "bitcoin": data.get("bitcoin", {}),
            "ethereum": data.get("ethereum", {}),
            "ripple": data.get("ripple", {})
        }
    except:
        return {
            "bitcoin": {"inr": "N/A", "usd": "N/A"},
            "ethereum": {"inr": "N/A", "usd": "N/A"},
            "ripple": {"inr": "N/A", "usd": "N/A"}
        }

@app.route('/')
def homepage():
    headlines = []
    timestamps = []
    for site in CRYPTO_SITES:
        try:
            res = requests.get(site, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            title = soup.title.string if soup.title else site
            date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            headlines.append(title)
            timestamps.append(date_time)
        except:
            headlines.append("[Error fetching: {}]".format(site))
            timestamps.append("N/A")

    sentiments = analyze_sentiment(headlines)
    data = list(zip(CRYPTO_SITES, headlines, sentiments, timestamps))
    prices = fetch_crypto_prices()
    return render_template("dashboard.html", data=data, prices=prices)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8501)