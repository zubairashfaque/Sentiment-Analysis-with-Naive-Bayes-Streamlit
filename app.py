import pandas as pd
import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from PIL import Image  # Import the Image module
nltk.download('punkt')
nltk.download("stopwords")
import math
from sklearn.model_selection import train_test_split
import base64
from plotly import graph_objs as go
import plotly.express as px
from io import BytesIO 
# Load the dataset
csv_file_path = '/data/train.csv'
df = pd.read_csv(csv_file_path)
df = df.drop(columns=['selected_text'])
df["text"] = df["text"].astype(str)
# Preprocessing function
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(tweet)
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words("english"))
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]
    return tokens

# Step 1: Split data by sentiment
def split_data_by_sentiment(data, sentiment):
    return data[data['sentiment'] == sentiment]['text'].tolist()

positive_tweets = split_data_by_sentiment(df, 'positive')
negative_tweets = split_data_by_sentiment(df, 'negative')
neutral_tweets = split_data_by_sentiment(df, 'neutral')

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Step 2: Calculate word counts for a given sentiment
def calculate_word_counts(tweets):
    word_count = defaultdict(int)
    for tweet in tweets:
        tokens = preprocess_tweet(tweet)
        for token in tokens:
            word_count[token] += 1
    return word_count

word_count_positive = calculate_word_counts(train_df[train_df['sentiment'] == 'positive']['text'])
word_count_negative = calculate_word_counts(train_df[train_df['sentiment'] == 'negative']['text'])
word_count_neutral = calculate_word_counts(train_df[train_df['sentiment'] == 'neutral']['text'])

# Display word counts
print("Word Counts - Positive Sentiment:")
print(word_count_positive)

print("\nWord Counts - Negative Sentiment:")
print(word_count_negative)

print("\nWord Counts - Neutral Sentiment:")
print(word_count_neutral)

# Step 3: Calculate likelihood using Laplacian smoothing
def calculate_likelihood(word_count, total_words, laplacian_smoothing=1):
    likelihood = {}
    vocabulary_size = len(word_count)

    for word, count in word_count.items():
        likelihood[word] = (count + laplacian_smoothing) / (total_words + laplacian_smoothing * vocabulary_size)

    return likelihood

total_positive_words = sum(word_count_positive.values())
total_negative_words = sum(word_count_negative.values())
total_neutral_words = sum(word_count_neutral.values())

likelihood_positive = calculate_likelihood(word_count_positive, total_positive_words)
likelihood_negative = calculate_likelihood(word_count_negative, total_negative_words)
likelihood_neutral = calculate_likelihood(word_count_neutral, total_neutral_words)

# Step 4: Calculate log likelihood
log_likelihood_positive = {word: math.log(prob) for word, prob in likelihood_positive.items()}
log_likelihood_negative = {word: math.log(prob) for word, prob in likelihood_negative.items()}
log_likelihood_neutral = {word: math.log(prob) for word, prob in likelihood_neutral.items()}

# Step 5: Calculate log prior
def calculate_log_prior(sentiment, data):
    return math.log(len(data[data['sentiment'] == sentiment]) / len(data))

log_prior_positive = calculate_log_prior('positive', df)
log_prior_negative = calculate_log_prior('negative', df)
log_prior_neutral = calculate_log_prior('neutral', df)

# Step 6: Classify a new tweet
def classify_tweet_with_scores(tweet, log_likelihood_positive, log_likelihood_negative, log_likelihood_neutral,
                   log_prior_positive, log_prior_negative, log_prior_neutral):
    tokens = preprocess_tweet(tweet)

    log_score_positive = log_prior_positive + sum([log_likelihood_positive.get(token, 0) for token in tokens])
    log_score_negative = log_prior_negative + sum([log_likelihood_negative.get(token, 0) for token in tokens])
    log_score_neutral = log_prior_neutral + sum([log_likelihood_neutral.get(token, 0) for token in tokens])

    sentiment_scores = {
        'positive': log_score_positive,
        'negative': log_score_negative,
        'neutral': log_score_neutral
    }

    predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    return predicted_sentiment, sentiment_scores

# Example test tweet

test_tweet = "I am not"
predicted_sentiment, sentiment_scores = classify_tweet_with_scores(
    test_tweet,
    log_likelihood_positive, log_likelihood_negative, log_likelihood_neutral,
    log_prior_positive, log_prior_negative, log_prior_neutral
)

print("Predicted sentiment:", predicted_sentiment)
print("Sentiment Scores:", sentiment_scores)


# Evaluate the model on the test set
correct_predictions = 0
total_predictions = len(test_df)

for index, row in test_df.iterrows():
    predicted_sentiment, sentiment_scores = classify_tweet_with_scores(
        row['text'], log_likelihood_positive, log_likelihood_negative, log_likelihood_neutral,
        log_prior_positive, log_prior_negative, log_prior_neutral
    )
    if predicted_sentiment == row['sentiment']:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print("Accuracy:", accuracy)


# Streamlit app
def main():

    # Add a banner image
    banner_image = Image.open("/images/sentimentanalysishotelgeneric-2048x803-1.jpg")  # Replace with your banner image filename
    st.image(banner_image, use_column_width=True)

    st.title("Sentiment Analysis with Naïve Bayes")

    # User input for text
    user_input = st.text_area("Enter a text:", "I love this product!", key="text_input")

    if st.button("Classify Sentiment", key="classify_button"):
        predicted_sentiment, sentiment_scores = classify_tweet_with_scores(
            user_input, log_likelihood_positive, log_likelihood_negative, log_likelihood_neutral,
            log_prior_positive, log_prior_negative, log_prior_neutral
        )
        image_size = (100, 100) 
        # Display an image based on the sentiment
        if predicted_sentiment == "positive":
            image = Image.open("/images/positive.jpg")  # Change to your positive image filename
            #image(image, caption="Positive Sentiment")
        elif predicted_sentiment == "negative":
            image = Image.open("/images/negative.jpg")  # Change to your negative image filename
            #image(image, caption="Negative Sentiment")
        else:
            image = Image.open("/images/neutral.jpg")   # Change to your neutral image filename
            #image(image, caption="Neutral Sentiment")
        resized_image = image.resize(image_size, Image.ANTIALIAS)
        #st.image(resized_image, caption=f"{predicted_sentiment.capitalize()} Sentiment", use_column_width=False)
        # Display resized image with markdown spacer for center alignment
        # Convert the image to base64
        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Display resized image with markdown spacer for center alignment
        st.markdown(
            f'<p align="center"><img src="data:image/png;base64,{img_str}" alt="{predicted_sentiment}" width="{image_size[0]}"></p>',
            unsafe_allow_html=True
        )
        # Display predicted sentiment in uppercase
        st.markdown(f'<p align="center"><b>{predicted_sentiment.upper()}</b></p>', unsafe_allow_html=True)

        # Display sentiment scores in a table
        scores_df = pd.DataFrame(sentiment_scores.items(), columns=['Sentiment', 'Score'])
        st.write("Sentiment Scores:")
        st.table(scores_df)


if __name__ == "__main__":
    main()