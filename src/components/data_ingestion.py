import os
import logging
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.npy')
    test_data_path: str = os.path.join('artifacts', 'test.npy')
    raw_data_path: str = r'C:\Users\OMEN\Malaria_Project\notebook\data\Train'

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load and preprocess the dataset
            X, y = self.load_dataset(self.ingestion_config.raw_data_path)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Save the data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            np.save(self.ingestion_config.train_data_path, {'X': X_train, 'y': y_train})
            np.save(self.ingestion_config.test_data_path, {'X': X_test, 'y': y_test})

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

    def load_dataset(self, directory, target_size=(150, 150)):
        images = []
        labels = []
        class_names = os.listdir(directory)
        for class_name in class_names:
            class_path = os.path.join(directory, class_name)
            class_label = 1 if class_name.lower() == 'uninfected' else 0
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(class_label)
        return np.array(images), np.array(labels)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Load and print information about the data
    train_data = np.load(train_data, allow_pickle=True).item()
    test_data = np.load(test_data, allow_pickle=True).item()

    print("Training set shape:", train_data['X'].shape)
    print("Test set shape:", test_data['X'].shape)

    print("Training set class distribution:")
    print("Uninfected:", np.sum(train_data['y'] == 1), "Infected:", np.sum(train_data['y'] == 0))
    print("Test set class distribution:")
    print("Uninfected:", np.sum(test_data['y'] == 1), "Infected:", np.sum(test_data['y'] == 0))