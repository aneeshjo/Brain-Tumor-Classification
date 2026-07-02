from pathlib import Path
import sys
import os
from datetime import datetime

import tensorflow as tf
from typing import List

from cnnClassifier.logger import logger
from cnnClassifier.exception import CustomException
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallbacks:

    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    def create_checkpoint_callback(self)-> tf.keras.callbacks.ModelCheckpoint:
        """
        Create and return a ModelCheckpoint callback.
        returns:
            tf.keras.callbacks.ModelCheckpoint: ModelCheckpoint callback.   
        """
        try:
            logger.info("Creating ModelCheckpoint callback...")
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.checkpoint_model_filepath,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1
            )
            logger.info("ModelCheckpoint callback created successfully.")
            return checkpoint_callback
        except Exception as e:
            logger.exception("Error while creating ModelCheckpoint callback.")
            raise CustomException(e, sys)

    def create_tensorboard_callback(self)-> tf.keras.callbacks.TensorBoard:
        """
        Create and return a TensorBoard callback.
        returns:
            tf.keras.callbacks.TensorBoard: TensorBoard callback.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            logger.info("Creating TensorBoard callback...")
            log_dir = Path(
                self.config.tensorboard_root_log_dir,
                f"fit_{timestamp}"
            )
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
            logger.info("TensorBoard callback created successfully.")
            return tensorboard_callback
        except Exception as e:
            logger.exception("Error while creating TensorBoard callback.")
            raise CustomException(e, sys)
        
    def get_callbacks(self)->List[tf.keras.callbacks.Callback]:
        """
        Get a list of callbacks for model training.
        returns:
            list: List of callbacks.
        """
        try:
            logger.info("Getting callbacks for model training...")
            checkpoint_callback = self.create_checkpoint_callback()
            tensorboard_callback = self.create_tensorboard_callback()
            callbacks = [checkpoint_callback, tensorboard_callback]
            logger.info("Callbacks for model training obtained successfully.")
            return callbacks
        except Exception as e:
            logger.exception("Error while getting callbacks for model training.")
            raise CustomException(e, sys)