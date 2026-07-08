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
            #easy to load image datasets directly from a folder structure. 
            train_dataset = tf.keras.utils.image_dataset_from_directory(
                # The directory where the training images are stored.
                directory=self.config.train_dir,
                # The target size to which all images will be resized.
                image_size=self.config.image_size,
                # The number of images to include in each batch.
                batch_size=self.config.batch_size,
                # Whether to shuffle the dataset after loading.
                shuffle=self.config.train_shuffle,
                # A seed for the random number generator to ensure reproducibility.
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
            # Keeps data in memory after the first epoch.

            # Speeds up subsequent epochs since images don’t need to be reloaded from disk.

            # Best when the dataset fits into memory.
            dataset = dataset.cache()
#     Loads the next batch while the current one is being processed.

# AUTOTUNE lets TensorFlow decide the optimal buffer size.

# Improves GPU utilization and reduces idle time.
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

            # Debug prints
            print("Training Class Names:", train_dataset.class_names)
            print("Testing Class Names:", test_dataset.class_names)

            train_dataset = self.optimize_dataset(train_dataset)
            test_dataset = self.optimize_dataset(test_dataset)

            logger.info("Datasets are ready for model training.")

            return train_dataset, test_dataset

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)