import sys

import tensorflow as tf

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger


class PrepareBaseModel:
    """
    Builds and saves the CNN architecture.
    """
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def build_model(self) -> tf.keras.Model:
        """
        Build the CNN architecture.

        Returns:
            tf.keras.Model: CNN model.
        """
        try:
            logger.info("Building CNN model...")

            model = tf.keras.Sequential()

            # Input Layer
            model.add(
                tf.keras.layers.Input(
                    shape=self.config.input_shape
                )
            )

            model.add(
                tf.keras.layers.Rescaling(1.0 / 255)
            )

            # Feature Extraction Block
            for filters in self.config.conv_filters:
                # Convolutional Layer
                model.add(
                    tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=self.config.kernel_size,
                        padding="same",
                        activation="relu"
                    )
                )
                #Batch Normalization Layer
                # 5
                model.add(
                    tf.keras.layers.MaxPooling2D(
                        pool_size=self.config.pool_size
                    )
                )

            # Classification Block
            model.add(
                tf.keras.layers.Flatten()
            )

            model.add(
                tf.keras.layers.Dense(
                    units=self.config.dense_units,
                    activation="relu"
                )
            )

            model.add(
                tf.keras.layers.Dropout(
                    rate=self.config.dropout_rate
                )
            )

            model.add(
                tf.keras.layers.Dense(
                    units=self.config.num_classes,
                    activation="softmax"
                )
            )

            logger.info("CNN model built successfully.")

            return model

        except Exception as e:
            logger.exception("Error while building CNN model.")
            raise CustomException(e, sys)

    def compile_model(self, model: tf.keras.Model) -> None:
        """
        Compile the CNN model.

        Args:
            model (tf.keras.Model): CNN model to compile.
        """
        try:
            logger.info("Compiling CNN model...")

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            logger.info("CNN model compiled successfully.")
        except Exception as e:
            logger.exception("Error while compiling CNN model.")
            raise CustomException(e, sys)