import os
from src.logger import logging
from src.exception import CustomException
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
    val_data_path: str = os.path.join('artifacts', 'val.npy')
    test_data_path: str = os.path.join('artifacts', 'test.npy')
    train_dir: str = r'C:\Users\OMEN\Malaria_Project\notebook\data\Train'
    test_dir: str = r'C:\Users\OMEN\Malaria_Project\notebook\data\Test'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load and preprocess the training dataset
            X_train, y_train = self.load_dataset(self.ingestion_config.train_dir, is_train=True)
            
            # Split the training data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Load and preprocess the test dataset
            X_test = self.load_dataset(self.ingestion_config.test_dir, is_train=False)

            # Save the data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            np.save(self.ingestion_config.train_data_path, {'X': X_train, 'y': y_train})
            np.save(self.ingestion_config.val_data_path, {'X': X_val, 'y': y_val})
            np.save(self.ingestion_config.test_data_path, {'X': X_test})

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

    def load_dataset(self, directory, target_size=(150, 150), is_train=True):
        images = []
        if is_train:
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
        else:
            for img_name in os.listdir(directory):
                img_path = os.path.join(directory, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
            return np.array(images)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, val_data, test_data = obj.initiate_data_ingestion()

    # Load and print information about the data
    train_data = np.load(train_data, allow_pickle=True).item()
    val_data = np.load(val_data, allow_pickle=True).item()
    test_data = np.load(test_data, allow_pickle=True).item()

    print("Training set shape:", train_data['X'].shape)
    print("Validation set shape:", val_data['X'].shape)
    print("Test set shape:", test_data['X'].shape)

    print("Training set class distribution:")
    print("Uninfected:", np.sum(train_data['y'] == 1), "Infected:", np.sum(train_data['y'] == 0))
    print("Validation set class distribution:")
    print("Uninfected:", np.sum(val_data['y'] == 1), "Infected:", np.sum(val_data['y'] == 0))
    print("Test set: No labels assigned")