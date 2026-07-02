from cnnClassifier.config.configuration import ConfigurationManager

config = ConfigurationManager()

base_model_config = config.get_prepare_base_model_config()

print(base_model_config)