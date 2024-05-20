import streamlit as st
import tensorflow as tf
import librosa
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

st.set_page_config(page_title="Evaluate Singing", page_icon="\🎤")

st.markdown("# Evaluate Singing")
st.sidebar.header("Evaluate Singing")

df = pd.read_csv(
    'https://storage.googleapis.com/harmonaize_dataset/harmonAIze_v3.csv')

st.write(
    """***Select details relavant to the singer***"""
)

gender = st.selectbox(
    "Select you gender",
    ("f", "m"))

st.write("Selected gender:", gender)

voice_type = st.selectbox(
    "Select voice type",
    ("soprano", "mezzo_soprano", "baritone", "tenor", "bass", "bass_baritone", "countertenor"))

st.write("Selected voice type:", voice_type)

st.write(
    """***Select details relavant to the exercise***"""
)

exercise_category = st.selectbox(
    "Select exercise category",
    ("arpeggios", "scales", "long"))

st.write("Selected category:", exercise_category)

exercise = st.selectbox(
    "Select exercise",
    ("belt", "breathy", "fast_forte", "fast_piano", "lip_trill", "slow_forte", "slow_piano", "straight", "vibrato", "vocal_fry", "forte", "messa", "pp", "trill", "trillo"))

st.write("Selected exercise:", exercise)

vowel = st.selectbox(
    "Select vowel",
    ("a", "e", "i", "o", "u"))

st.write("Selected vowel:", vowel)

categorical_features = df[[
    'gender', 'exercise_category', 'exercise', 'vowel', 'voice_type']]

# One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_encoded = encoder.fit_transform(categorical_features)

st.write(
    """***Upload an audio recording of you performing the exercise***"""
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write("filename:", uploaded_file.name)
    st.write('Preview uploaded voice recording')
    st.audio(uploaded_file, format="audio/mpeg", loop=False)

model = tf.keras.models.load_model('./model_weights/harmonAIze_weights.h5')


def get_mse(original, reconstruction):
    return np.mean(np.square(original - reconstruction), axis=1)


def load_wav(filename):
    st.write('File loaded for prediction')
    audio, sr = librosa.load(filename, sr=None)
    return audio, sr


def extract_mel_features(audio, sr, n_mels=128):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(S, ref=np.max)
    flattened_mel_db = mel_db.flatten()

    return flattened_mel_db[:128]


def predict():
    audio, sr = load_wav(uploaded_file)
    mel_features_new = extract_mel_features(audio, sr)
    rec_data_attributes = {'gender': [gender], 'exercise_category': [
        exercise_category], 'exercise': [exercise], 'vowel': [vowel], 'voice_type': [voice_type]}
    new_categorical_df = pd.DataFrame(rec_data_attributes)
    new_categorical_encoded = encoder.transform(new_categorical_df)
    flattened_mel_features_new = mel_features_new.flatten()
    expected_mel_feature_length = 33 - new_categorical_encoded.shape[1]
    flattened_mel_features_new = flattened_mel_features_new[:expected_mel_feature_length].reshape(
        1, -1)
    combined_features_new = np.hstack(
        (new_categorical_encoded, flattened_mel_features_new))
    reconstructions = model.predict(combined_features_new)
    mse = get_mse(combined_features_new, reconstructions)
    cutoff = 0.039078482566601014
    is_anomaly = mse > cutoff
    st.write('Error in your performance', mse)
    st.write('Is anomaly:', is_anomaly)


if st.button('Evaluate your performance'):
    predict()
