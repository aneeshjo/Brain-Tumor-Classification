import sys

from cnnClassifier.logger import logger
from cnnClassifier.exception import CustomException
from cnnClassifier.pipeline.training_pipeline import TrainingPipeline


def main():
    """
    Entry point of the Brain Tumor Classification application.
    """
    try:
        logger.info("=" * 80)
        logger.info("Brain Tumor Classification Application Started")
        logger.info("=" * 80)

        training_pipeline = TrainingPipeline()

        history = training_pipeline.run_pipeline()

        logger.info("Training pipeline executed successfully.")

        return history

    except Exception as e:
        logger.exception("Application execution failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()