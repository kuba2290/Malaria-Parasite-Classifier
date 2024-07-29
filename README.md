# Malaria Parasite Classifier

## Project Overview
This project implements a machine learning model to classify blood cell images as either infected with malaria parasites or uninfected. It uses a Convolutional Neural Network (CNN) to analyze microscopic images of blood cells and predict the presence of malaria parasites.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Data](#data)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)

## Installation

1. Clone the repository:
https://github.com/kuba2290/Malaria-Parasite-Classifier.git
cd malaria-parasite-classifier

2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the required packages:
pip install -r requirements.txt

## Project Structure
malaria-parasite-classifier/
│
├── data/
│   ├── Train/
│   └── Test/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   │
│   ├── utils.py
│   └── exception.py
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── artifacts/
│
├── logs/
│
├── tests/
│
├── config/
│   └── config.yaml
│
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore

## Usage

1. Data Ingestion:
python src/components/data_ingestion.py

2. Model Training:
python src/pipeline/train_pipeline.py

3. Make Predictions:
python src/pipeline/predict_pipeline.py

## Data
The dataset consists of microscopic images of blood cells, both infected with malaria parasites and uninfected. The data is split into training and test sets.

- You can find the dataset used at https://drive.google.com/drive/folders/1Fshh206Vmbds8VFuGUTZlGk4uT2c7nE6?usp=drive_link https://drive.google.com/drive/folders/16pX7lrv6lgPZZoXHiJ5YV2LcG1qrQI30?usp=drive_link
- Train set: 1598 images
- Test set: 1316 images

## Model Architecture
The model uses a Convolutional Neural Network (CNN) with the following architecture:
- 3 Convolutional layers with ReLU activation
- Max Pooling layers
- Flatten layer
- Dense layer with ReLU activation
- Dropout for regularization
- Output layer with sigmoid activation for binary classification

## Training
The model is trained using:
- Optimizer: Adam
- Loss function: Binary Cross-Entropy
- Metrics: Accuracy
- Regularization: L2 regularization and Dropout

## Evaluation
The model's performance is evaluated using:
- Accuracy

## Results
(Include your model's performance metrics here once you have them)

## Future Improvements
- Implement data augmentation to increase dataset diversity
- Experiment with transfer learning using pre-trained models
- Explore ensemble methods for improved accuracy
- Develop a web interface for easy use by medical professionals

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.