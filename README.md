# Music Genre Classification 🎶

Music classification has become an essential tool in today's digital landscape, enabling users to accurately categorize and discover songs based on their unique sound characteristics. Traditional genre classification methods often limit a song to a single genre, whereas our advanced deep-learning model provides a more nuanced breakdown, identifying multiple genres present in a track with a percentage-based probability.

## Features

- **Multi-genre classification:** Predicts the probability distribution across multiple genres for a given song.
- **User-friendly web apps:** Choose between a modern Streamlit dashboard or a classic Flask web interface.
- **High accuracy:** Model trained on Mel spectrograms using a Convolutional Neural Network (CNN).
- **Chunk-based analysis:** Audio is split into 10-second chunks for robust classification.

## Project Structure

```
.
├── app.py                # Flask web app backend
├── app.html              # Flask frontend (template)
├── app.css               # Flask frontend (styles)
├── musicsort.py          # Streamlit web app
├── source_code.ipynb     # Model training notebook
```

## How It Works

Our project leverages Convolutional Neural Networks (CNNs) to analyze Mel spectrograms generated from audio files. The process includes:
1. **Audio Preprocessing:** Uploaded MP3 files are split into 10-second chunks.
2. **Mel Spectrogram Generation:** Each chunk is converted into a Mel spectrogram image.
3. **Genre Prediction:** The trained CNN model predicts genre probabilities for each chunk.
4. **Results Presentation:** The app displays the overall genre distribution for the song.

## Web Apps

### 1. Streamlit App

- File: [`musicsort.py`](musicsort.py)
- Run with:
  ```sh
  streamlit run musicsort.py
  ```
- Features a dashboard, project info, and prediction page with audio upload.

### 2. Flask App

- Backend: [`app.py`](app.py)
- Frontend: [`app.html`](app.html), [`app.css`](app.css)
- Run with:
  ```sh
  python app.py
  ```
- Visit [http://localhost:5000](http://localhost:5000) in your browser.

## Model Training

- See [`source_code.ipynb`](source_code.ipynb) for model architecture and training code.
- Trained model weights are stored in the [`h5_files/`](h5_files/) directory.

## Requirements

- Python 3.7+
- TensorFlow / Keras
- librosa
- numpy
- matplotlib
- Pillow
- scikit-learn
- Flask
- Streamlit

Install dependencies with:
```sh
pip install -r requirements.txt
```
*(Create `requirements.txt` as needed.)*

## Usage

- **Streamlit:** Upload an MP3 and get genre probabilities instantly.
- **Flask:** Use the web form to upload an MP3 and view the predicted genre and distribution.

## Credits

- Developed by [S.Sidhardha Reddy]
- Model trained on Mel spectrograms of music tracks.

---
## Contact

sidhardhareddy73@gmail.com
