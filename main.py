import sys

from cnnClassifier.logger import logger
from cnnClassifier.exception import CustomException
from cnnClassifier.pipeline.training_pipeline import TrainingPipeline
from cnnClassifier.pipeline.evaluation_pipeline import EvaluationPipeline


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

        logger.info("=" * 50)
        logger.info("Starting Evaluation Pipeline")
        logger.info("=" * 50)

        evaluation_pipeline = EvaluationPipeline()

        results = evaluation_pipeline.run_pipeline()

        logger.info("=" * 50)
        logger.info("CNN Pipeline Completed Successfully")
        logger.info("=" * 50)

        return history

    except Exception as e:
        logger.exception("Application execution failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()