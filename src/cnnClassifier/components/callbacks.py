import os
import sys
from pathlib import Path
from datetime import datetime

import tensorflow as tf

from cnnClassifier.entity.config_entity import PrepareCallbacksConfig
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger


class PrepareCallback:
    """
    This class is responsible for creating all TensorFlow callbacks
    required during model training.
    """

    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    def create_tensorboard_callback(self):
        """
        Creates TensorBoard callback for visualizing training logs.
        """

        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            tensorboard_log_dir = os.path.join(
                self.config.tensorboard_root_log_dir,
                f"tb_logs_{timestamp}"
            )

            logger.info(f"TensorBoard log directory: {tensorboard_log_dir}")

            return tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_dir
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def create_checkpoint_callback(self):
        """
        Creates ModelCheckpoint callback.
        Saves only the best model.
        """

        try:

            logger.info(
                f"Checkpoint path: {self.config.checkpoint_model_filepath}"
            )

            return tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.checkpoint_model_filepath,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def create_early_stopping_callback(self):
        """
        Stops training if validation accuracy
        doesn't improve for several epochs.
        """

        try:

            return tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def create_reduce_lr_callback(self):
        """
        Reduces learning rate when validation
        accuracy stops improving.
        """

        try:

            return tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=3,
                verbose=1
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def get_callbacks(self):
        """
        Returns all callbacks as a list.
        """

        logger.info("Preparing TensorFlow callbacks.")

        callback_list = [
            self.create_tensorboard_callback(),
            self.create_checkpoint_callback(),
            self.create_early_stopping_callback(),
            self.create_reduce_lr_callback(),
        ]

        logger.info("Callbacks created successfully.")

        return callback_list