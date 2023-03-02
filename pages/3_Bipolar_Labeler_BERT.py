import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("Joom/bert-base-uncased-NisadiBipolar")
    return tokenizer,model


tokenizer,model = get_model()

st.markdown("""
    
#### Bipolar Labeling Tool, 

  This tool utilizes a deep learning BERT model to analyze tweet descriptions and predict signs of bipolar disorder. By leveraging the power of machine learning, our tool can analyze large datasets of tweets and identify patterns and language that are indicative of bipolar disorder.
In addition to its predictive capabilities, This tool can also be used to label tweets when creating a new dataset. This is particularly useful for researchers and analysts who are looking to build their own datasets for studying mental health conditions such as bipolar disorder.
  """)


user_input = st.text_area('Copy Paste a tweet and click predict')
button = st.button("Predict")

d = {
    
  1:'No signs of Bipolar Disorder',
  0:'Possible Signs of Bipolar Disorder'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=256,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    # st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])