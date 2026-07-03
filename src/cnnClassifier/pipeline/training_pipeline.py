import sys
import tensorflow as tf

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_transformation import DataTransformation
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks
from cnnClassifier.components.model_trainer import ModelTrainer
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logger

class TrainingPipeline:

    def __init__(self):
        pass
    def run_pipeline(self)->tf.keras.callbacks.History:
        try:
            logger.info("Starting training pipeline...")
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            transformation = DataTransformation(config=data_transformation_config)
            train_dataset, validation_dataset = transformation.get_datasets()

            base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=base_model_config)
            model = prepare_base_model.build_model()
            prepare_base_model.compile_model(model=model)

            callbacks_config = config.get_prepare_callbacks_config()
            prepare_callbacks = PrepareCallbacks(config=callbacks_config)
            callbacks = prepare_callbacks.get_callbacks()

            model_training_config = config.get_model_training_config()
            model_trainer = ModelTrainer(config=model_training_config)
            history = model_trainer.train(
                model=model,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                callbacks=callbacks
            )

            logger.info("Training pipeline completed successfully.")
            return history
        except Exception as e:
            logger.exception("Error occurred in training pipeline.")
            raise CustomException(e, sys)