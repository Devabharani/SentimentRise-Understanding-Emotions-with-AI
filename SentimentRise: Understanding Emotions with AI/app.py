import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (do this only once)
nltk.download('vader_lexicon')

# Create a SentimentIntensityAnalyzer object
analyzer = SentimentIntensityAnalyzer()

# Function to classify a sentence and provide an explanation
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

# Input from the user
user_input = input("Enter a sentence: ")

# Classify the input sentence
sentiment, explanation = classify_sentiment(user_input)

# Display the result
print(f"The sentence is classified as {sentiment}.")
print(explanation)
