import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
#Load the LSTM model

model_lstm=load_model('next_word_lstm.h5')
model_gru=load_model('next_word_gru.h5')

with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)
def predict_next_word(model,tokenizer,text,maxlen):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=maxlen:
        token_list=token_list[-(maxlen-1):]
    token_list=pad_sequences([token_list],maxlen=maxlen-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None
#StreamLIT App
st.title('Next Word Prediction with LSTM and GRU ')
input_text=st.text_input("Enter the amount of Words","To be or not to be")

if st.button("Predict"):
    max_seq_len=model_lstm.input_shape[1]+1
    next_word_lstm=predict_next_word(model_lstm,tokenizer,input_text,max_seq_len)
    st.write(f"Next Word in LSTM:{next_word_lstm}")
    next_word_gru=predict_next_word(model_gru,tokenizer,input_text,max_seq_len)
    st.write(f"Next Word in GRU:{next_word_gru}")
