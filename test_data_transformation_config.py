from cnnClassifier.config.configuration import ConfigurationManager

config = ConfigurationManager()

transformation_config = config.get_data_transformation_config()

print(transformation_config)