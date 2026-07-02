from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks

config = ConfigurationManager()

callback_config = config.get_prepare_callbacks_config()

prepare_callbacks = PrepareCallbacks(callback_config)

checkpoint_callback = prepare_callbacks.create_checkpoint_callback()

print(checkpoint_callback)