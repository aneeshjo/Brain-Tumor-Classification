import sys

import tensorflow as tf

from cnnClassifier.entity.config_entity import DataTransformationConfig
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config

    def create_train_dataset(self):
        try:
            logger.info("Creating training dataset...")
            train_dataset=tf.keras.utils.image_dataset_from_directory(
                directory=self.config.train_dir,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            logger.info("Training dataset created successfully.")

            return train_dataset

        except Exception as e:

            raise CustomException(e, sys)
        
    def create_validation_dataset(self):

        try:
            logger.info("Creating validation dataset...")
            validation_dataset = tf.keras.utils.image_dataset_from_directory(
                directory=self.config.test_dir,

                image_size=self.config.image_size,

                batch_size=self.config.batch_size,

                shuffle=False,
            )

            logger.info("Validation dataset created successfully.")

            return validation_dataset
        except Exception as e:

            raise CustomException(e, sys)
    
    def optimize_dataset(self, dataset):

        dataset = dataset.cache()

        dataset = dataset.prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        return dataset
    def get_datasets(self):

        train_dataset = self.create_train_dataset()

        validation_dataset = self.create_validation_dataset()

        train_dataset = train_dataset.shuffle(self.config.shuffle_buffer_size)

        train_dataset = self.optimize_dataset(train_dataset)

        validation_dataset = self.optimize_dataset(validation_dataset)

        return train_dataset, validation_dataset