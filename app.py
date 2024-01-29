import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_msgs(msg):
    # converting text into lower
    msg = msg.lower()

    # word tokenizing, split the words
    msg = nltk.word_tokenize(msg)

    # removing special characters
    y = []
    for i in msg:
        if i.isalnum():
            y.append(i)

    msg = y[:]
    y.clear()

    # removing stopwords and punctuation
    for i in msg:
        if i not in stopwords.words('english'):
            y.append(i)

    msg = y[:]
    y.clear()

    # stemming the words
    ps = PorterStemmer()
    for i in msg:
        y.append(ps.stem(i))
    
    return " ".join(y)



st.title("Spam Email classifier")

input_sms = st.text_input("Enter the Message")

if st.button("Predict"):

    # proprocess
    transformed_sms = transform_msgs(input_sms)
    # vectorizer
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]
    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")