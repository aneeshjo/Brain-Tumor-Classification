import os
import sys
from pathlib import Path

from cnnClassifier.entity.config_entity import DataValidationConfig
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger

class DataValidation:
    """
    Validates the dataset before training.
    """
    EXPECTED_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
    ]

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset_exists(self)->bool:
        """
        Validate that the dataset exists.
        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        try:
            dataset_path = self.config.unzip_data_dir

            logger.info("Starting dataset existence validation.")

            if not dataset_path.exists():
                logger.error(f"Dataset path {dataset_path} does not exist.")
                return False

            logger.info("Dataset existence validation successful.")

            return True

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)
    def validate_training_folder(self) -> bool:
        """
        Validate that the training folder exists.
        Returns:
            bool: True if the training folder exists, False otherwise.
        """
        try:
            training_path = self.config.unzip_data_dir / "Training"

            logger.info("Starting training folder validation.")

            if not training_path.exists():
                logger.error(f"Training path {training_path} does not exist.")
                return False

            logger.info("Training folder validation successful.")

            return True

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)
    def validate_testing_folder(self) -> bool:
        """
        Validate that the testing folder exists.
        Returns:
            bool: True if the testing folder exists, False otherwise.
        """
        try:
            testing_path = self.config.unzip_data_dir / "Testing"

            logger.info("Starting testing folder validation.")

            if not testing_path.exists():
                logger.error(f"Testing path {testing_path} does not exist.")
                return False

            logger.info("Testing folder validation successful.")

            return True

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)  

    def validate_class_folders(self) -> bool:
        """
        Validate that all expected class folders exist
        in both Training and Testing directories.
        Returns:
            bool: True if all expected class folders exist, False otherwise.
        """

        try:
            training_path = self.config.unzip_data_dir / "Training"
            testing_path = self.config.unzip_data_dir / "Testing"

            for class_name in self.EXPECTED_CLASSES:
                if not (training_path / class_name).exists():
                    logger.error(f"Class folder {class_name} does not exist in Training directory.")
                    return False

                if not (testing_path / class_name).exists():
                    logger.error(f"Class folder {class_name} does not exist in Testing directory.")
                    return False
            logger.info("All expected class folders exist in both Training and Testing directories.")
            return True
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
 

    def validate_dataset_structure(self) -> bool:
        """
        Validate that the dataset structure exists.
        Returns:
            bool: True if the dataset structure is valid, False otherwise.
        """
        try:
            logger.info("Starting dataset structure validation.")

            if not self.validate_dataset_exists():
                return False

            if not self.validate_training_folder():
                return False

            if not self.validate_testing_folder():
                return False
            if not self.validate_class_folders():
                return False

            logger.info("Dataset structure validation successful.")

            return True

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)
        