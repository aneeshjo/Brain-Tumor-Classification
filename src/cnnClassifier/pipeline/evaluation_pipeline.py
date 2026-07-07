from cnnClassifier.components.data_transformation import DataTransformation
from cnnClassifier.components.model_evaluation import ModelEvaluation
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger
import sys

class EvaluationPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):
        """
        Run the evaluation pipeline to evaluate the trained model on the test dataset.
        Returns:
            dict: Evaluation metrics including classification report and confusion matrix.
        """
        try:
            logger.info("Starting evaluation pipeline...")
            config_manager = ConfigurationManager()
            data_transformation_config = (
            config_manager.get_data_transformation_config()
            )

            transformation = DataTransformation(
                config=data_transformation_config
            ) 

            _, test_dataset = (
                transformation.get_datasets()
            )
            evaluation_config = config_manager.get_model_evaluation_config()
            evaluator = ModelEvaluation(config=evaluation_config)
            results = evaluator.evaluate(
                test_dataset
            )
            logger.info("Evaluation pipeline completed successfully.")
            return results
        except Exception as e:
            logger.exception("Error occurred in evaluation pipeline.")
            raise CustomException(e, sys)