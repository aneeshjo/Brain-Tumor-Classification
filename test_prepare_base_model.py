from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel

config = ConfigurationManager()

prepare_model_config = config.get_prepare_base_model_config()

prepare_model = PrepareBaseModel(prepare_model_config)

model = prepare_model.build_model()

prepare_model.compile_model(model)

model.summary()
print(model.optimizer)