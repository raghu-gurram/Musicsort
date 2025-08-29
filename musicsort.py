import os
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
from collections import Counter
from pathlib import Path
import base64

# Define class names
CLASS_NAMES = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "EDM",
    4: "Hip-Hop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock"
}

# Function to split audio into 10-second chunks
def split_audio(file_path, chunk_duration=10):
    y, sr = librosa.load(file_path, sr=None)
    chunk_length = chunk_duration * sr
    return [y[i:i + chunk_length] for i in range(0, len(y), chunk_length)], sr

# Function to create and save Mel spectrogram
def create_mel_spectrogram(chunk, sr, output_folder, file_prefix, segment_index):
    os.makedirs(output_folder, exist_ok=True)

    # Generate Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot and save spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis')
    plt.axis('off')

    # Save as image
    temp_image_path = os.path.join(output_folder, f"temp_{file_prefix}_{segment_index}.png")
    plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0, dpi=72)
    plt.close()

    # Resize to 512x512
    with Image.open(temp_image_path) as img:
        img = img.resize((512, 512))
        final_path = os.path.join(output_folder, f"{file_prefix}_mel_spectrogram_{segment_index}.png")
        img.save(final_path)
    os.remove(temp_image_path)


# Function to process audio and classify
def process_and_classify(audio_path, model_path):
    model = load_model(model_path)
    output_dir = "output_spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    chunks, sr = split_audio(audio_path)
    all_probabilities = []
    all_class_indices = []

    for i, chunk in enumerate(chunks):
        if len(chunk) < sr * 10:
            continue
        file_prefix = f"chunk_{i + 1}"
        create_mel_spectrogram(chunk, sr, output_dir, file_prefix, i + 1)

        mel_img_path = os.path.join(output_dir, f"{file_prefix}_mel_spectrogram_{i + 1}.png")
        mel_img = Image.open(mel_img_path).convert('RGB')
        img_array = np.array(mel_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        probabilities = model.predict(img_array)
        predicted_index = np.argmax(probabilities[0])
        all_probabilities.append(probabilities[0])
        all_class_indices.append(predicted_index)

    class_counts = Counter(all_class_indices)
    total_chunks = sum(class_counts.values())
    overall_percentages = {
        CLASS_NAMES[cls]: (count / total_chunks) * 100 for cls, count in class_counts.items()
    }
    sorted_percentages = sorted(overall_percentages.items(), key=lambda x: x[1], reverse=True)
    return sorted_percentages, sorted_percentages[0][0]

# Streamlit App
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

if app_mode == "Home":
    st.title("Welcome to Music Genre Classification 🎶")
    st.markdown("Discover your music genre with ease!")
    def get_base64_image(image_path):
        image_data = Path(image_path).read_bytes()
        return base64.b64encode(image_data).decode()
    
    def add_bg_from_local(image_file):
        encoded_image = get_base64_image(image_file)
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/image.png;base64,{encoded_image});
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Set the background image
    add_bg_from_local("image/image3.jpeg")

elif app_mode == "About Project":
    st.title("About the Project")
    st.markdown("""

Overview

Music classification has become an essential tool in today's digital landscape, enabling users to accurately categorize and discover songs based on their unique sound characteristics. Traditional genre classification methods often limit a song to a single genre, whereas our advanced deep-learning model provides a more nuanced breakdown, identifying multiple genres present in a track with a percentage-based probability.

How It Works

Our project leverages Convolutional Neural Networks (CNNs) to analyze Mel spectrograms generated from audio files. The process includes:

Audio Preprocessing: The uploaded audio file is split into 10-second chunks for better classification accuracy.

Mel Spectrogram Generation: Each chunk is transformed into a visual spectrogram representation, preserving its frequency and temporal characteristics.

Genre Prediction: The trained deep-learning model processes the spectrogram images and predicts the likelihood of each genre being present in the audio.

Results Presentation: Instead of assigning just one genre, the model outputs a percentage-based distribution, providing insights into the blended nature of modern music.

Features

Multi-genre classification: Unlike conventional classifiers, our model provides multiple genre probabilities rather than a single label.

User-friendly interface: The web application, powered by Streamlit, allows easy interaction with the model, making it accessible for everyone.

High Accuracy: Our deep-learning model has been trained on an extensive dataset to ensure high precision in music classification.

Enhanced Music Discovery: By analyzing hidden genres within a song, our tool helps users find music that matches their preferences more accurately.

Applications

Playlist Optimization: Users can curate more refined and personalized playlists.

Music Recommendation: Streaming platforms can enhance their recommendation algorithms by incorporating multi-genre classification.

Music Production: Artists and producers can analyze genre influences in their compositions for better creative insights.

With this project, we aim to revolutionize music classification and help users explore their favorite genres with greater depth and accuracy. Experience the future of AI-driven music discovery today!
    """)

elif app_mode == "Prediction":
    st.title("Genre Prediction")
    uploaded_mp3 = st.file_uploader("Upload an MP3 file", type=["mp3"])
    h5_model_path = "h5_files/model_5_90(76.8, 1.7).h5"

    if uploaded_mp3:
        file_path = f"temp_{uploaded_mp3.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_mp3.getbuffer())

        st.audio(file_path)

        if st.button("Predict"):
            with st.spinner("Analyzing audio..."):
                results, top_class = process_and_classify(file_path, h5_model_path)
                st.success(f"The top predicted genre is: {top_class}")
                st.markdown("### Genre Distribution:")
                for genre, perc in results:
                    st.markdown(f"- **{genre}**: {perc:.2f}%")
