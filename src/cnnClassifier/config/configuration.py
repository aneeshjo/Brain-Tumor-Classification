from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import (
     read_yaml,
     create_directories
)

from src.cnnClassifier.entity.config_entity import (
    DataIngestionConfig
)
class ConfigurationManager:
    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        self.config=read_yaml(config_file_path)
        self.params=read_yaml(params_file_path)
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