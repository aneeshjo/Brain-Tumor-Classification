from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path

    train_dir: Path

    test_dir: Path

    image_size: tuple

    batch_size: int

    shuffle_buffer_size:int

@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration required for Data Validation.
    """

    root_dir: Path
    status_file: Path
    unzip_data_dir: Path
    expected_classes: list  # List of expected class names
    allowed_extensions: set  # List of allowed file extensions for images

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path

    train_dir: Path

    test_dir: Path

    image_size: tuple

    batch_size: int

    seed: int

    train_shuffle: bool

    test_shuffle: bool

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path

    model_path: Path

    input_shape: tuple

    num_classes: int

    conv_filters: list

    kernel_size: tuple

    pool_size: tuple

    dense_units: int

    learning_rate: float

    dropout_rate: float
@dataclass(frozen=True)
class PrepareCallbacksConfig:

    root_dir: Path

    checkpoint_model_filepath: Path

    tensorboard_root_log_dir: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path

    trained_model_path: Path

    epochs: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    evaluation_file_path: Path
    target_names: list

@dataclass(frozen=True)
class ModelPredictionConfig:

    root_dir: Path

    model_path: Path

    image_size: list

    class_names: list