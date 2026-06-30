import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

def create_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Create a simple CNN model.
    """
    model=Sequential(
        [
            Conv2D(
                filters=32,
                kernel_size=(3,3),
                activation="relu",
                input_shape=input_shape
            ),
            MaxPooling2D(
                pool_size=(2,2)
            ),
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
            ),

            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
            ),

            Flatten(),

            Dense(
                128,
                activation="relu",
            ),

            Dropout(0.5),

            Dense(
                num_classes,
                activation="softmax",
            )

        ]
    )

    return model