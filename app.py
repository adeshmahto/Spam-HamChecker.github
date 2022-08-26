import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
stop_words = set(stopwords.words('english'))

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # separating words

    y = []
    for i in text:
        if i.isalnum():  # it is used to check numeric or not  or it used to clear out the special char
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Email/SMS spam Classifier")
input_sms=st.text_area("Enter the Message")
if st.button('Predict'):
    # 1. preprocess
    transformed_sms=transform_text(input_sms)
    # 2. vectorize
    vector_input=tfidf.transform([transformed_sms])
    # 3. predict
    result=model.predict(vector_input)[0]
    # 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

