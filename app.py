import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# Pvectorizer = TfidfVectorizer(stop_words='english', max_features=500)

# Create file uploader and define accepted file types
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# If a file was uploaded, read the contents into a Pandas DataFrame
if uploaded_file is not None:
    pdata = pd.read_csv(uploaded_file).dropna()
    # pdata.columns = pdata.columns.astype(str)
    pdata['timestamp'] = pd.to_datetime(pdata['timestamp'])
    pdata['hour'] = pdata['timestamp'].dt.hour
    pdata['weekday'] = pdata['timestamp'].dt.weekday
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)

# Vectorize the text data
    text_features = vectorizer.fit_transform(pdata['tweet'])

    pfeature_matrix = pd.concat([pd.DataFrame(text_features.toarray()), pdata[['hour', 'weekday']]], axis=1)
    pfeature_matrix['label'] = pdata['bp_label']
    pft = pfeature_matrix.dropna()
    # Use the trained model to predict the chances of a new patient having bipolar disorder
    new_tweet_history_vec = pft.drop(columns=['label'])
    inputvect = new_tweet_history_vec.columns.astype(int)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # new_tweet_history_vec = vectorizer.transform([new_tweet_history])
    prob = model.predict_proba(inputvect)[0][1]
    st.write("Probability of having bipolar disorder:", prob)
