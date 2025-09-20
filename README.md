# Audio Diagnosis for Washing Machine Sounds

## Overview
This project focuses on analyzing audio recordings from washing machines to identify whether they are operating normally or exhibiting signs of abnormal behavior. By leveraging sound analysis techniques, we aim to provide insights into the operational status of washing machines, which can help in preventive maintenance and troubleshooting.

## Objectives
- To develop a system that can differentiate between normal and abnormal washing machine sounds.
- To utilize audio signal processing techniques for feature extraction.
- To implement machine learning algorithms for classification of sounds.

## Features
- **Real-time audio analysis**: Monitor washing machine sounds in real-time.
- **Sound classification**: Identify normal and abnormal operation through sound patterns.
- **User-friendly interface**: Easy-to-use interface for users to upload audio recordings and receive analysis results.

## Methodology
1. **Data Collection**: Gather audio samples from various washing machines under different operating conditions (normal and abnormal).
2. **Preprocessing**: Clean and preprocess the audio files for analysis, including noise reduction and normalization.
3. **Feature Extraction**: Extract relevant features from the audio signals using techniques such as Mel-frequency cepstral coefficients (MFCCs) and spectral analysis.
4. **Model Training**: Train machine learning models using labeled audio data to classify the sounds. Possible models include Support Vector Machines (SVM), Random Forests, and Neural Networks.
5. **Evaluation**: Test the model on unseen data to evaluate its performance and accuracy.

## Installation
To set up the project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/AryanSaxenaa/AudioDiagnosis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd AudioDiagnosis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Upload your audio recording through the provided interface.
3. Analyze the results to determine the operation status of your washing machine.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Special thanks to the contributors and everyone who provided audio samples for this project.
- References to the libraries and tools used in this project can be found in the `requirements.txt` file.