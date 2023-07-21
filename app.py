import streamlit as st
import pickle
import emoji
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = emoji.demojize(text)
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for j in text:
        if j not in stopwords.words('english') and j not in string.punctuation:
            y.append(j)
            
    text = y[:]
    y.clear()
    
    for k in text:
        y.append(ps.stem(k))
        
    return " ".join(y)


# Load pickle file
tfidf = pickle.load(open('model/vectorizer.pkl', 'rb'))
model = pickle.load(open('model/model.pkl', 'rb'))

st.title("Cyber-Bullying Comment Classifier")

input_comment = st.text_area("Enter the comment")

if st.button('Predict'):


    transformed_text = transform_text(input_comment)

    vector_input = tfidf.transform([transformed_text])

    result = model.predict(vector_input)[0]
    
    if result ==0:
        st.header('Result : Religious Cyber-Bully Found')

    if result ==1:
        st.header('Result : Gender Cyber-Bully Found')

    if result==2:
        st.header('Result : Age Cyber-Bully Found')

    if result ==3:
        st.header('Result : Ethnicity Cyber-Bully Found')

    if result ==4:
        st.header('Result : No Cyber-Bully Found')

    else:
        st.header('Result : Others Cyber-Bully Found')