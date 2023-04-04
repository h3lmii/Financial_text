from pickle import load
import tensorflow as tf
from keras.utils import pad_sequences
import streamlit as st
from tensorflow.keras.models import load_model

def generate_text(input_text, num_gen_words, model, max_len,tokenizer):
    for _ in range(num_gen_words):

        output_text = ""

        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        padded_encoded = pad_sequences([encoded_text], maxlen=max_len-1, padding='pre')
        predicted = model.predict(padded_encoded, verbose=0).argmax()
        
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_text = word
                break
        input_text += " "+output_text
    return input_text.title()

with open('tokenizer', 'rb') as f:
    loaded_tokenizer = load(f)

loaded_model=load_model('generation_model.h5',compile=False)


max_len=44
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}

       .st-b7 {
    color: white;
    font-size=10px;
}

    textarea {
        font-size: 1.5rem !important;
    }
    

       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title('FINANCIAL REPORTS TEXT GENERATION')
col1, col2 = st.columns(2)
with col1:
    text = st.text_input('Enter some text to be generated 👇')

with col2:
    num_gen_words = st.slider('Number of word generated', 0, 30, 10)

if st.button('GENERATE'):
    txt = st.text_area('THE GENERATED TEXT',generate_text(text, num_gen_words, loaded_model, max_len,loaded_tokenizer))







