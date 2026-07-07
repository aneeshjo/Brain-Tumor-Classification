import json
import sys
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from cnnClassifier.entity.config_entity import ModelEvaluationConfig
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger

class ModelPrediction:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def load_model(self) -> tf.keras.Model:
        
        """
        Load a trained Keras model from the specified path.

        Args:
            model_path (str): Path to the saved Keras model.

        Returns:
            tf.keras.Model: The loaded Keras model.
        """
       

        try:
            logger.info(f"Loading the trained model from {self.config.model_path}...")
            model = tf.keras.models.load_model(self.config.model_path)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.exception("Error occurred while loading the model.")
            raise CustomException(e, sys)
        
    def preprocess_image(
    self,
    image_path: str
) -> np.ndarray:
        """
        Preprocess an image for model prediction.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: The preprocessed image.
        """
        try:
            logger.info(f"Preprocessing image from {image_path}...")
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=tuple(self.config.image_size))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0  # Normalize the image
            image = np.expand_dims(image, axis=0)
            logger.info("Image preprocessing completed.")
            return image
        except Exception as e:
            logger.exception("Error occurred while preprocessing the image.")
            raise CustomException(e, sys)

    def predict(
    self,
    image_path: str
) -> dict:
        """
        Predict the class of an image using the trained model.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: A dictionary containing the predicted class and confidence score.
        """
        try:
            logger.info(f"Starting prediction for image: {image_path}")
            model = self.load_model()
            preprocessed_image = self.preprocess_image(image_path)
            logger.info("Making prediction...")
            predictions = model.predict(preprocessed_image, verbose=0)
            print("Raw Predictions:")
            print(predictions)

            for i, class_name in enumerate(self.config.class_names):
                print(f"{class_name}: {predictions[0][i]:.4f}")
            print("Predicted Index:", np.argmax(predictions, axis=1)[0])
            print("Class Names:", self.config.class_names)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = float(np.max(predictions))*100
            predicted_class_name = self.config.class_names[predicted_class_index]
            logger.info(f"Prediction completed: {predicted_class_name} with confidence {confidence_score:.2f}")
            return {
                "predicted_class": predicted_class_name,
                "confidence_score": round(confidence_score, 2)
            }
        except Exception as e:
            logger.exception("Error occurred during prediction.")
            raise CustomException(e, sys)       
       