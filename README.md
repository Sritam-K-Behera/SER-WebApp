# Speech Emotion Recognition (SER) App

A web application for recognizing emotions from speech using a TensorFlow Lite model, designed for microcontrollers like the Arduino Nano BLE33. The app is trained on the RAVDESS dataset and can identify seven different emotions.

## Features

- **Audio Upload**: Upload .wav files for emotion recognition.
- **Real-time Recording**: Record and analyze audio in real-time.
- **Feature Extraction**: Extracts features such as Zero Crossing Rate (ZCR), Root Mean Square Energy (RMSE), and Mel-Frequency Cepstral Coefficients (MFCCs).
- **Data Augmentation**: Enhances feature extraction using noise addition and pitch shifting.
- **Emotion Prediction**: Standardizes features and uses a pre-trained TensorFlow model to predict emotions.

## Recognized Emotions

- Angry😡
- Disgust😖
- Fear😱
- Happy😊
- Neutral😐
- Sad😥
- Surprise😮

## How to Use

1. **Upload an Audio File**: Click the "Upload file for Speech Emotion Recognition" button and select a .wav file.
2. **Record Audio**: Click the "Start Recording for Emotion Prediction" button to record a new audio sample.
3. **View Results**: The predicted emotion will be displayed after analysis.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ompathak23/SER-App.git
    cd SER-App
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Folder Structure

- **audio/**: Directory for storing uploaded and recorded audio files.
- **models/**: Directory containing the pre-trained TensorFlow model (`SER.hdf5`).
- **app.py**: Main application script.
- **requirements.txt**: List of required Python packages.
- **README.md**: Project description and instructions.

## Data Augmentation Functions

- **noise(data, random=False, rate=0.035, threshold=0.075)**: Adds noise to the audio data.
- **pitch(data, sampling_rate, pitch_factor=0.7, random=False)**: Changes the pitch of the audio data.

## Feature Extraction Functions

- **zcr(data, frame_length=2048, hop_length=512)**: Computes Zero Crossing Rate.
- **rmse(data, frame_length=2048, hop_length=512)**: Computes Root Mean Square Energy.
- **mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True)**: Computes Mel-Frequency Cepstral Coefficients.
- **extract_features(data, sr, frame_length=2048, hop_length=512)**: Extracts a combination of features from the audio data.

## Logging and File Management

- **log_file(txt=None)**: Logs file processing details.
- **save_audio(file)**: Saves uploaded audio files and manages storage by clearing old files.


## Acknowledgements

- The model is trained on the [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) dataset.
- Made with 🖤 by SAS.
