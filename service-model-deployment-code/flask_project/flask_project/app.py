from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'secret-key'

# Load DistilBERT model and tokenizer
MODEL_PATH = '/srv/models/distilbert_sentiment_model'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Sentiment labels mapping
LABELS = {0: "negative", 1: "neutral", 2: "positive"}

def classify_reviews_in_batches(reviews, batch_size=32):
    """
    Classify reviews in batches to improve performance.
    """
    sentiments = []

    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i+batch_size].tolist()
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).tolist()
            sentiments.extend([LABELS[p] for p in predictions])

    return sentiments

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files or request.files['csv_file'].filename == '':
            flash('No file selected. Please upload a CSV file.')
            return redirect(request.url)

        file = request.files['csv_file']

        if not file.filename.endswith('.csv'):
            flash('Invalid file format. Please upload a valid CSV file.')
            return redirect(request.url)

        try:
            data = pd.read_csv(file)
            if 'reviews.text' not in data.columns:
                flash("The CSV file must contain the 'reviews.text' column.")
                return redirect(request.url)

            data['reviews.text'] = data['reviews.text'].fillna("No review provided").astype(str)

            # Classify reviews in batches
            data['sentiment'] = classify_reviews_in_batches(data['reviews.text'])

            output_file = os.path.join('/tmp', 'classified_reviews.csv')
            data.to_csv(output_file, index=False)

            flash('File processed successfully! You can download the updated dataset below.')
            return render_template('index.html', download_link=output_file)

        except Exception as e:
            flash(f"Error processing the file: {e}")
            return redirect(request.url)

    return render_template('index.html')

@app.route('/download')
def download_file():
    path = request.args.get('file')
    if path:
        return send_file(path, as_attachment=True, download_name='classified_reviews.csv')
    else:
        flash("No file available for download.")
        return redirect(url_for('index'))
    

def classify_comment(comment):
    """
    Classify a single user comment using the DistilBERT model.
    """
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()  # Get the predicted label
    return LABELS[prediction]

@app.route('/home')
def home():
    """
    Home page with links to upload a CSV or classify a comment.
    """
    return render_template('home.html')

@app.route('/classify-comment', methods=['GET', 'POST'])
def classify_comment_page():
    """
    Page for user to enter a comment and classify its sentiment.
    """
    if request.method == 'POST':
        comment = request.form.get('comment')
        if not comment or comment.strip() == "":
            flash('Please enter a valid comment.')
            return redirect(request.url)
        
        sentiment = classify_comment(comment)
        return render_template('classify_comment.html', comment=comment, sentiment=sentiment)

    # If it's a GET request, just show the form
    return render_template('classify_comment.html')


if __name__ == '__main__':
    app.run(debug=True)
