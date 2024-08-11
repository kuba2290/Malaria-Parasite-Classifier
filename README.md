# Malaria Detection using Deep Learning

## Project Overview
This project implements a deep learning model to detect malaria parasites in cell images. Using Convolutional Neural Networks (CNNs), the model classifies images as either 'infected' or 'uninfected', aiding in the rapid and accurate diagnosis of malaria.

## Features
- Data preprocessing and augmentation
- Custom CNN model architecture
- Training with early stopping and learning rate reduction
- Model evaluation on test set
- Visualization of training history and confusion matrix

## Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

For a complete list of requirements, see `requirements.txt`.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/kuba2290/Malaria-Parasite-Classifier.git
   cd malaria-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset
The project uses the Malaria Cell Images Dataset. You can download it from [Kaggle](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).

After downloading, organize the data as follows:
```
notebook/
└── data/
    ├── Train/
    │   ├── infected/
    │   └── uninfected/
    └── Test/
```

## Usage

1. Ensure your data is organized as described above.

2. Run the main script:
   ```
   python malaria_detection.py
   ```

3. The script will:
   - Load and preprocess the data
   - Train the model
   - Display training history
   - Evaluate the model on the test set
   - Show a confusion matrix of the results

## Model Architecture
The model uses a custom CNN architecture with the following key features:
- Multiple convolutional and max pooling layers
- Dropout for regularization
- Binary classification output

For detailed architecture, refer to the `create_model()` function in the script.

## Results
The model achieves 60% accuracy on the test set. Detailed metrics including precision, recall, and F1 score are printed after evaluation.



## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
 Twitter - [https://twitter.com/kubasmide]

Project Link: [https://github.com/kuba2290/Malaria-Parasite-Classifier]

## Acknowledgements
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)