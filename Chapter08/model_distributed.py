import os
os.environ["KERAS_BACKEND"] = "jax" 

import keras
from keras import layers
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, set_distribution
import jax

import numpy as np
from tensorflow import data as tf_data
devices = jax.devices("gpu")
print(devices)
mesh = DeviceMesh(shape=(1, 2), axis_names=["data", "model"], devices=devices)
layout_map = LayoutMap(mesh)
layout_map["d1/kernel"] = (None, "model")
layout_map["d1/bias"] = ("model",)
layout_map["d2/output"] = ("data", None)

model_parallel = ModelParallel(layout_map=layout_map, batch_dim_name="data")

set_distribution(model_parallel)

x = np.random.normal(size=(128, 28, 28, 1))
y = np.random.normal(size=(128, 10))
dataset = tf_data.Dataset.from_tensor_slices((x, y)).batch(16)

inputs = layers.Input(shape=(28, 28, 1))
y = layers.Flatten()(inputs)
y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
y = layers.Dropout(0.4)(y)
y = layers.Dense(units=10, activation="softmax")(y)
model = keras.Model(inputs=inputs, outputs=y)

model.compile(loss="mse")
model.fit(dataset, epochs=3)
print(model.evaluate(dataset))
