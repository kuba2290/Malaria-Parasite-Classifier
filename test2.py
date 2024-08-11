import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)


def load_dataset(directory, target_size=(224, 224)):
    """Load and preprocess the dataset."""
    images = []
    labels = []
    class_names = os.listdir(directory)
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        class_label = 1 if class_name == 'uninfected' else 0
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=target_size
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(class_label)
    return np.array(images), np.array(labels)


def create_model():
    """Create and compile the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3),
               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_and_display(model, img_path, target_size=(224, 224)):
    """Predict and display results for a single image."""
    try:
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=target_size
        )
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)

        result = "uninfected" if prediction[0][0] > 0.5 else "infected"
        print(f"Predicted: {result}")

        return prediction[0][0]
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return None


def main():
    # Load and split the data
    train_dir = r'C:\Users\OMEN\Malaria_Project\notebook\data\Train'
    X, y = load_dataset(train_dir)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Check class distribution
    train_infected = np.sum(y_train == 0)
    train_uninfected = np.sum(y_train == 1)
    val_infected = np.sum(y_val == 0)
    val_uninfected = np.sum(y_val == 1)

    print(f"Training set - Infected: {train_infected}, "
          f"Uninfected: {train_uninfected}")
    print(f"Validation set - Infected: {val_infected}, "
          f"Uninfected: {val_uninfected}")

    # Create and compile the model
    model = create_model()
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr]
    )

    # Plot training history
    plot_training_history(history)

    # Test the model
    test_dir = r'C:\Users\OMEN\Malaria_Project\notebook\data\Test'
    all_images = os.listdir(test_dir)
    selected_images = random.sample(all_images, min(100, len(all_images)))

    results = []
    true_labels = []
    for img_name in selected_images:
        img_path = os.path.join(test_dir, img_name)
        print(f"\nProcessing image: {img_name}")
        probability = predict_and_display(model, img_path)
        if probability is not None:
            results.append({'image': img_name, 'probability': probability})
            true_label = 1 if "uninfected" in img_name.lower() else 0
            true_labels.append(true_label)

    # Display summary of results
    print("\nSummary of Results:")
    for result in results:
        print(f"Image: {result['image']}, "
              f"Probability of being uninfected: {result['probability']:.4f}")

    # Calculate overall statistics
    predicted_labels = [1 if r['probability'] > 0.5 else 0 for r in results]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"\nTotal images processed: {len(results)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == "__main__":
    main()