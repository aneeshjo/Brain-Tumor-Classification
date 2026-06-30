from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion

config = ConfigurationManager()

data_config = config.get_data_ingestion_config()

ingestion = DataIngestion(data_config)

ingestion.copy_data()

ingestion.extract_zip_file()