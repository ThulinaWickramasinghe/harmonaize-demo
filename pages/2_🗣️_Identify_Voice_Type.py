import streamlit as st
import tensorflow as tf
# from numpy import random

# x = random.randint(0,10)

st.set_page_config(page_title="Evaluate Singing", page_icon="\🎤")

st.markdown("# Evaluate Singing")
st.sidebar.header("Evaluate Singing")
st.write(
    """Upload an audio recording of you performing the exercise"""
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write("filename:", uploaded_file.name)
    st.write('Preview uploaded voice recording')
    st.audio(uploaded_file, format="audio/mpeg", loop=False)

model = tf.keras.models.load_model('./model_weights/voice_typ_final_vers2.h5')
st.write(model.summary())