import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")


# Create file uploader and define accepted file types
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# If a file was uploaded, read the contents into a Pandas DataFrame
if uploaded_file is not None:
    pdata = pd.read_csv(uploaded_file).dropna()
    # pdata.columns = pdata.columns.astype(str)
    pdata = pdata.drop(columns=['bp_label'])
    pdata['timestamp'] = pd.to_datetime(pdata['timestamp'])
    pdata['hour'] = pdata['timestamp'].dt.hour
    pdata['weekday'] = pdata['timestamp'].dt.weekday
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    textTweet = pdata['tweet']
    text = " ".join(tweet for tweet in textTweet)
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)
    plt.figure(figsize=(8,8))
    y= plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(y)


    >>> fig, ax = plt.subplots()
>>> ax.scatter([1, 2, 3], [1, 2, 3])
>>>    ... other plotting actions ...
>>> st.pyplot(fig)

# Vectorize the text data
    text_features = vectorizer.fit_transform(pdata['tweet'])
    x = pd.DataFrame(text_features.toarray())
    x.columns = x.columns.astype(str)

    pfeature_matrix = pd.concat([x, pdata[['hour', 'weekday']]], axis=1)
    # pfeature_matrix['label'] = pdata['bp_label']
    pft = pfeature_matrix.dropna()
    # pft =pft.columns.astype(str)
    # Use the trained model to predict the chances of a new patient having bipolar disorder
    new_tweet_history_vec = pft
    # inputvect = new_tweet_history_vec.columns.astype(float)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # new_tweet_history_vec = vectorizer.transform([new_tweet_history])
    prob = model.predict_proba(new_tweet_history_vec)[0][1]
    st.write("The chances of This user having bipolar disorder:", "{:.2f}%".format(prob*100))









# st.markdown("[Colab NoteBook Used for Traning Machine Learning]:  https://lk.linkedin.com/in/nisandi-jayasuriya-294327194?trk=people_directory&original_referer=)")

