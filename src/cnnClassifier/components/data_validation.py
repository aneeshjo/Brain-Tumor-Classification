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
   

    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.validation_results = {}

        self.image_counts = {
            "Training": {},
            "Testing": {}
        }

    
    @property
    def training_path(self):
        return self.config.unzip_data_dir / "Training"


    @property
    def testing_path(self):
        return self.config.unzip_data_dir / "Testing"

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
                self.validation_results["Dataset Exists"] = "FAIL"
                return False

            logger.info("Dataset existence validation successful.")
            self.validation_results["Dataset Exists"] = "PASS"
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
            # training_path = self.config.unzip_data_dir / "Training"

            logger.info("Starting training folder validation.")

            if not self.training_path.exists():
                logger.error(f"Training path {self.training_path} does not exist.")
                self.validation_results["Training Folder Exists"] = "FAIL"
                return False

            logger.info("Training folder validation successful.")
            self.validation_results["Training Folder Exists"] = "PASS"

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
            # testing_path = self.config.unzip_data_dir / "Testing"

            logger.info("Starting testing folder validation.")

            if not self.testing_path.exists():
                logger.error(f"Testing path {self.testing_path} does not exist.")
                self.validation_results["Testing Folder Exists"] = "FAIL"   
                return False

            logger.info("Testing folder validation successful.")
            self.validation_results["Testing Folder Exists"] = "PASS"
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
            # training_path = self.config.unzip_data_dir / "Training"
            # testing_path = self.config.unzip_data_dir / "Testing"

            for class_name in self.config.expected_classes:
                if not (self.training_path / class_name).exists():
                    logger.error(f"Class folder {class_name} does not exist in Training directory.")
                    self.validation_results["Class Folders Exist"] = "FAIL"
                    return False

                if not (self.testing_path / class_name).exists():
                    logger.error(f"Class folder {class_name} does not exist in Testing directory.")
                    self.validation_results["Class Folders Exist"] = "FAIL"
                    return False
            logger.info("All expected class folders exist in both Training and Testing directories.")
            self.validation_results["Class Folders Exist"] = "PASS"
            return True
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
    
    def validate_images(self, folder_path: Path) -> bool:
        """
    Validate that all files inside the given folder
    have supported image extensions.
    """
        try:
            logger.info(f"Starting image validation in folder: {folder_path}")

            for class_folder in folder_path.iterdir():
                if not class_folder.is_dir():
                    continue  # Skip if it's not a directory

                for image_file in class_folder.iterdir():
                    if not image_file.is_file():
                        continue  # Skip if it's not a file
                    if image_file.suffix.lower() not in self.config.allowed_extensions:
                        logger.error(f"Unsupported image extension found: {image_file}")
                        self.validation_results["Image Extensions"] = "FAIL"
                        return False
            logger.info(f"Image validation successful in folder: {folder_path}")
            self.validation_results["Image Extensions"] = "PASS"
            return True
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def validate_empty_folders(self, folder_path: Path) -> bool:
        """
        Validate that every class folder contains at least one image.
        Returns:
            bool: True if all class folders contain at least one image, False otherwise.
        """

        try:

            logger.info(f"Checking empty folders in: {folder_path}")

            for class_folder in folder_path.iterdir():
                if not class_folder.is_dir():
                    continue  # Skip if it's not a directory
                
                image_files = [
                                image_file
                                for image_file in class_folder.iterdir()
                                if (
                                        image_file.is_file()
                                        and image_file.suffix.lower()
                                        in self.config.allowed_extensions
                                    )
                                ]
                self.image_counts[folder_path.name][class_folder.name] = len(image_files)
                if not image_files:
                    logger.error(f"Class folder {class_folder.name} is empty or contains no valid images.")
                    self.validation_results["Empty Folders"] = "FAIL"
                    return False
            logger.info(f"All class folders in {folder_path.name} contain at least one valid image.")
            self.validation_results["Empty Folders"] = "PASS"
            return True
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
    def write_validation_report(self) -> None:
        """
        Write the data validation report to the status file.
        """

        try:
            logger.info("Generating validation report.")

            # Ensure the directory exists
            self.config.status_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            report_lines = []

            # ============================
            # Report Header
            # ============================
            report_lines.append("=" * 60)
            report_lines.append("                DATA VALIDATION REPORT")
            report_lines.append("=" * 60)
            report_lines.append("")

            # ============================
            # Validation Results
            # ============================
            report_lines.append("Validation Results")
            report_lines.append("-" * 60)

            for key, value in self.validation_results.items():
                report_lines.append(f"{key:<30} : {value}")

            report_lines.append("")

            # ============================
            # Image Counts
            # ============================
            report_lines.append("=" * 60)
            report_lines.append("IMAGE COUNTS")
            report_lines.append("=" * 60)

            for dataset_type, class_counts in self.image_counts.items():

                report_lines.append("")
                report_lines.append(f"{dataset_type}")
                report_lines.append("-" * 60)

                for class_name, count in class_counts.items():
                    report_lines.append(
                        f"{class_name:<20} : {count}"
                    )

            report_lines.append("")

            # ============================
            # Overall Status
            # ============================
            overall_status = (
                "SUCCESS"
                if all(value == "PASS" for value in self.validation_results.values())
                else "FAILED"
            )

            report_lines.append("=" * 60)
            report_lines.append(f"Overall Status : {overall_status}")
            report_lines.append("=" * 60)

            # ============================
            # Write Report
            # ============================
            with open(self.config.status_file, "w") as file:
                file.write("\n".join(report_lines))

            logger.info(
                f"Validation report saved to: {self.config.status_file}"
            )

            logger.info(
            f"Validation report saved to: {self.config.status_file.resolve()}"
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
    def validate_dataset(self) -> bool:
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
            if not self.validate_images(self.training_path):
                return False
            if not self.validate_images(self.testing_path):
                return False
            if not self.validate_empty_folders(self.training_path):
                return False
            if not self.validate_empty_folders(self.testing_path):
                return False
            self.write_validation_report()

            logger.info("Dataset structure validation successful.")

            return True

        except Exception as e:

            logger.error(e)

            raise CustomException(e, sys)
        