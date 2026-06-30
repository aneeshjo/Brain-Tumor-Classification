from cnnClassifier.config.configuration import ConfigurationManager

config = ConfigurationManager()

validation_config = config.get_data_validation_config()

print(validation_config)