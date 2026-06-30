from cnnClassifier.config.configuration import ConfigurationManager

config = ConfigurationManager()

data_config = config.get_data_ingestion_config()

print(data_config)