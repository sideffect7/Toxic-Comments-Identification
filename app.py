import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np
import pandas as pd

st.title('Toxic Comment Detection')
label={0:'toxic', 1:'severe_toxic', 2:'obscene', 3:'threat', 4:'insult',
       5:'identity_hate'}

df = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv")

def vectorized_text(text):
    x=df['comment_text']
    y=df[df.columns[2:]].values
    max_no_of_words=200000
    vectorizer = TextVectorization(max_tokens=max_no_of_words,
                           output_sequence_length=1800,
                           output_mode='int')
    vectorizer.adapt(x.values)
    text=vectorizer(text)
    return text

def load_model(model_path):
    model = tf.keras.models.load_model(model_path,compile=False)
    return model

def predict(text):
    model=load_model('toxicity.h5')
    pred = model.predict(np.expand_dims(text,0))
    return pred

textInput=st.text_input('Enter your Toxic comment here')


if textInput is not "":
    textInput=vectorized_text(textInput)
    prediction=label[np.argmax(predict(textInput))]
    st.write(prediction)