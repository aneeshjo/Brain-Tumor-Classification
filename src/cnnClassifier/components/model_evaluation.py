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


class ModelEvaluation:

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
        logger.info(f"Loading the trained model from {self.config.model_path}...")

        try:
            model = tf.keras.models.load_model(self.config.model_path)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.exception("Error occurred while loading the model.")
            raise CustomException(e, sys)
        
    def generate_predictions(
            self,
            model: tf.keras.Model,
            test_dataset: tf.data.Dataset
        ) -> tuple[list[int], list[int]]:
        """
        Generate predictions using the trained model on the test dataset.

        Args:
            model (tf.keras.Model): The trained Keras model.
            test_dataset (tf.data.Dataset): The test dataset."""
        
        try:
            logger.info("Generating predictions on the test dataset...")
            y_true = []
            y_pred = []
            for images, labels in test_dataset:
                y_true.extend(labels.numpy())
              
                predictions = model.predict(images, verbose=0)
                predicted_labels = np.argmax(
                    predictions,
                    axis=1
                )               
                y_pred.extend(predicted_labels)

            

            logger.info(
    f"Generated predictions for {len(y_true)} samples."
            )
            return y_true, y_pred
            
            
        except Exception as e:
            logger.exception("Error occurred while generating predictions.")
            raise CustomException(e, sys)
    def calculate_metrics(
            self,
            y_true: list,
            y_pred: list
        ) -> dict:
        """
        Calculate evaluation metrics.

        Returns:
            dict: Contains the classification report and confusion matrix.
        """
        try:
            logger.info("Calculating evaluation metrics...")
            cm= confusion_matrix(y_true, y_pred)
            report = classification_report(
                y_true,
                y_pred,
                target_names=self.config.target_names,

                output_dict=True)
            results = {
                "classification_report": report,
                "confusion_matrix": cm.tolist()
                }
            logger.info("Evaluation metrics calculated successfully.")
            return results
            

        except Exception as e:
            logger.exception("Error occurred while calculating evaluation metrics.")
            raise CustomException(e, sys)
        
    def save_results(self, results: dict) -> None:
        """
        Save evaluation results to a JSON file.

        Args:
            results (dict): Evaluation results to be saved.
        """
        try:
            logger.info("Saving evaluation results to JSON file...")
            with open(self.config.evaluation_file_path, 'w') as file:
                json.dump(results, file, indent=4)

                logger.info(
                    f"Evaluation results saved to {self.config.evaluation_file_path}")

        except Exception as e:
            logger.exception(
                "Error occurred while saving evaluation results."
            )
            raise CustomException(e, sys)

    def evaluate(self,test_dataset: tf.data.Dataset) -> dict:
        """
        Evaluate the trained model on the test dataset.

        Args:
            test_dataset (tf.data.Dataset): The test dataset.

        Returns:
    t       uple: Ground truth labels and predicted labels.
        """
        try:
            logger.info("Loading the trained model for evaluation...")
            model = tf.keras.models.load_model(self.config.model_path)
            logger.info("Model loaded successfully.")

            logger.info("Evaluating the model on the test dataset...")


            y_true, y_pred = self.generate_predictions(model, test_dataset)

            results = self.calculate_metrics(y_true=y_true, y_pred=y_pred)

            self.save_results(results)
            logger.info("Model evaluation completed successfully.")
            return results

        except Exception as e:
            logger.exception("Error occurred during model evaluation.")
            raise CustomException(e, sys) 