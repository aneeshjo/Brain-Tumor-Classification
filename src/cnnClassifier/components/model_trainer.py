import tensorflow as tf
import sys
from cnnClassifier.entity.config_entity import ModelTrainingConfig

from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def train(
    self,
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    callbacks: list[tf.keras.callbacks.Callback]
    )-> tf.keras.callbacks.History:
        """
        Train the CNN model.

        Args:
            model (tf.keras.Model): The CNN model to be trained.
            train_dataset (tf.data.Dataset): The training dataset.
            validation_dataset (tf.data.Dataset): The validation dataset.
            callbacks (list): List of callbacks to be used during training. 

        Returns:
            tf.keras.callbacks.History: Training history.
        """
        try:
            logger.info("Starting model training...")
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=self.config.epochs,
                callbacks=callbacks
            )
            logger.info("Model training completed successfully.")
            return history
        except Exception as e:
            logger.exception("Error during model training.")
            raise CustomException(e, sys)