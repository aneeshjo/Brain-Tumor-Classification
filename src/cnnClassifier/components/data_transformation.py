import sys

import tensorflow as tf

from cnnClassifier.entity.config_entity import DataTransformationConfig
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger


class DataTransformation:
    """
    Handles loading and optimization of TensorFlow datasets
    for training and testing.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_train_dataset(self) -> tf.data.Dataset:
        """
        Load the training dataset from the training directory.

        Returns:
            tf.data.Dataset: Training dataset.
        """
        try:
            logger.info("Loading training dataset...")

            train_dataset = tf.keras.utils.image_dataset_from_directory(
                directory=self.config.train_dir,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                shuffle=self.config.train_shuffle,
                seed=self.config.seed
            )

            logger.info("Training dataset loaded successfully.")

            return train_dataset

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def load_test_dataset(self) -> tf.data.Dataset:
        """
        Load the testing dataset from the testing directory.

        Returns:
            tf.data.Dataset: Testing dataset.
        """
        try:
            logger.info("Loading testing dataset...")

            test_dataset = tf.keras.utils.image_dataset_from_directory(
                directory=self.config.test_dir,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                shuffle=self.config.test_shuffle,
                seed=self.config.seed
            )

            logger.info("Testing dataset loaded successfully.")

            return test_dataset

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def optimize_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Optimize the dataset for better training performance.

        Steps:
        1. Cache the dataset.
        2. Prefetch batches while the model is training.

        Args:
            dataset (tf.data.Dataset): Input dataset.

        Returns:
            tf.data.Dataset: Optimized dataset.
        """
        try:
            logger.info("Optimizing dataset...")

            dataset = dataset.cache()

            dataset = dataset.prefetch(
                buffer_size=tf.data.AUTOTUNE
            )

            logger.info("Dataset optimized successfully.")

            return dataset

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def get_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load and optimize both training and testing datasets.

        Returns:
            tuple:
                (
                    train_dataset,
                    test_dataset
                )
        """
        try:
            logger.info("Preparing TensorFlow datasets...")

            train_dataset = self.load_train_dataset()
            test_dataset = self.load_test_dataset()

            train_dataset = self.optimize_dataset(train_dataset)
            test_dataset = self.optimize_dataset(test_dataset)

            logger.info("Datasets are ready for model training.")

            return train_dataset, test_dataset

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)