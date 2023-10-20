import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, render_template, request

nltk.download('vader_lexicon')

# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    user_input = request.form['sentence']
    sentiment, explanation = classify_sentiment(user_input)
    return render_template('result.html', sentence=user_input, sentiment=sentiment, explanation=explanation)

def classify_sentiment(sentence):
    sentiment_scores = analyzer.polarity_scores(sentence)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment = "good"
        explanation = "The sentiment of the sentence is positive."
    elif compound_score <= -0.05:
        sentiment = "bad"
        explanation = "The sentiment of the sentence is negative."
    else:
        sentiment = "neutral"
        explanation = "The sentiment of the sentence is neutral."

    return sentiment, explanation

if __name__ == '__main__':
    app.run(debug=True)
