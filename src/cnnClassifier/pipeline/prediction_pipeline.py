import sys

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_prediction import ModelPrediction
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger

class PredictionPipeline:

    def __init__(self):
        pass

    def run_pipeline(
    self,
    image_path: str
        ) -> dict:
        """
        Run the prediction pipeline for a given image.

        Args:
            image_path (str): Path to the image file."""
        try:
            logger.info("Starting prediction pipeline...")
            config_manager = ConfigurationManager()

            prediction_config = (
                config_manager.get_model_prediction_config()
            )

            predictor = ModelPrediction(
                config=prediction_config
            )

            results = predictor.predict(
                image_path=image_path
            )
            logger.info("Prediction pipeline completed successfully.")
            return results
        except Exception as e:
            logger.error("Error occurred while initializing prediction pipeline.")
            raise CustomException(e, sys)   
        
