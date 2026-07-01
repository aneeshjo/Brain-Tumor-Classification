from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_validation import DataValidation

config = ConfigurationManager()

validation_config = config.get_data_validation_config()

validator = DataValidation(validation_config)

status = validator.validate_dataset()

print(f"Validation Status: {status}")