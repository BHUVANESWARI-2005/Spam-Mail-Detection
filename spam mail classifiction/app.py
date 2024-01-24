import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and model using pickle
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # More training data and labels
    train_data = [
        "Congratulations! You've won a free vacation. Claim your prize now.",
        "Limited time offer! Get 50% off on our exclusive products.",
        "Your account has been compromised. Please reset your password.",
        "Important meeting tomorrow at 10 AM. Don't forget to attend.",
        "Click the link to claim your prize. You are our lucky winner!",
        "Double your money in just one week. Join our investment program.",
        "URGENT: Your bank account has been suspended. Please click the link to verify your details.",
        "Special discount for our valued customers. Use code SAVE20 at checkout.",
        "Reminder: Your subscription is about to expire. Renew now for uninterrupted service.",
        "You have been selected for a job interview. Please confirm your attendance.",
    ]

    train_labels = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]  # 1 for spam, 0 for not spam

    # Fit the TF-IDF vectorizer and the model on the training data
    tfidf.fit(train_data)
    model.fit(tfidf.transform(train_data), train_labels)

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
