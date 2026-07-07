import os
import shutil
import sys
import zipfile

from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def copy_data(self):
        try:
            source=self.config.source_URL
            destination=self.config.local_data_file

            logger.info(f"Copying dataset from {source}")

            shutil.copy(source, destination) # Copy the dataset from the source URL to the local data file path.

            logger.info("Dataset copied successfully.")
        except Exception as e:
            raise CustomException(e,sys)
        
    def extract_zip_file(self):
        try:
            unzip_path=self.config.unzip_dir
            zip_path = self.config.local_data_file

            logger.info("Extracting dataset...")

            with zipfile.ZipFile(zip_path,"r") as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info("Dataset extracted successfully.")
        except Exception as e:
            raise CustomException(e,sys)