from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import (
     read_yaml,
     create_directories
)

from src.cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelPredictionConfig
)
class ConfigurationManager:
    def __init__(self,
        config_file_path=CONFIG_FILE_PATH,
        params_file_path=PARAMS_FILE_PATH
        ):
        self.config=read_yaml(config_file_path)
        self.params=read_yaml(params_file_path)
        # Create the artifacts root directory if it doesn't exist
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([Path(config.root_dir)])
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,

            local_data_file=Path(config.local_data_file),

            unzip_dir=Path(config.unzip_dir),

        )

        return data_ingestion_config
    
    def get_data_transformation_config(self)->DataTransformationConfig:
        """
        Creates Data Transformation Configuration.
        Returns:
            DataTransformationConfig: Configuration for data transformation.
        """

        # Get the data transformation configuration from the main config
        config=self.config.data_transformation
        # Create the root directory for data transformation if it doesn't exist-
        # artifacts/data_transformation
        create_directories([Path(config.root_dir)])

        data_transformation_config = DataTransformationConfig(

            root_dir=Path(config.root_dir),

            train_dir=Path(config.train_dir),

            test_dir=Path(config.test_dir),

            image_size=tuple(self.params.IMAGE_SIZE),

            batch_size=self.params.BATCH_SIZE,

            # shuffle_buffer_size=self.params.SHUFFLE_BUFFER_SIZE,

            seed=self.params.SEED,
            train_shuffle=self.params.TRAIN_SHUFFLE,
            test_shuffle=self.params.TEST_SHUFFLE

        )

        return data_transformation_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Creates Data Validation Configuration.
        """

        config = self.config.data_validation

        create_directories([Path(config.root_dir)])

        data_validation_config = DataValidationConfig(

            root_dir=Path(config.root_dir),

            status_file=Path(config.status_file),

            unzip_data_dir=Path(config.unzip_data_dir),
            expected_classes=config.expected_classes,
            allowed_extensions=config.allowed_extensions

        )

        return data_validation_config
    
    def get_prepare_base_model_config(self)->PrepareBaseModelConfig:
        config=self.config.prepare_base_model
        create_directories([Path(config.root_dir)])
        prepare_base_model_config=PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            input_shape=tuple(self.params.INPUT_SHAPE),
            num_classes=self.params.NUM_CLASSES,
            conv_filters=self.params.CONV_FILTERS,
            kernel_size=tuple(self.params.KERNEL_SIZE),
            learning_rate=self.params.LEARNING_RATE,
            pool_size=tuple(self.params.POOL_SIZE),
            dense_units=self.params.DENSE_UNITS,
            dropout_rate=self.params.DROPOUT_RATE
        )
        return prepare_base_model_config
    
    def get_prepare_callbacks_config(self)->PrepareCallbacksConfig:
        config=self.config.prepare_callbacks
        create_directories([Path(config.root_dir)])
        prepare_callbacks_config=PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            patience=self.params.PATIENCE
        )
        return prepare_callbacks_config
    
    def get_model_training_config(self)->ModelTrainingConfig:
        config=self.config.model_training
        create_directories([Path(config.root_dir)])
        model_training_config=ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            epochs=self.params.EPOCHS
        )
        return model_training_config
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([Path(config.root_dir)])
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            evaluation_file_path=Path(config.evaluation_file_path),
            target_names=config.target_names
        )
        return model_evaluation_config
    
    def get_model_prediction_config(self) -> ModelPredictionConfig:

        config = self.config.model_prediction
        create_directories([Path(config.root_dir)])
        model_prediction_config = ModelPredictionConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            image_size=list(self.params.IMAGE_SIZE),
            class_names=config.class_names
        )
        return model_prediction_config
        