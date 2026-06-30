from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_transformation import DataTransformation

config = ConfigurationManager()

transformation_config = config.get_data_transformation_config()

transformation = DataTransformation(transformation_config)

train_ds, val_ds = transformation.get_datasets()

print(train_ds)

print(val_ds)