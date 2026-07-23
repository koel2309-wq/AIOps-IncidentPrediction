import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

X = np.random.random((100, 10, 10)).astype(np.float32)
y = np.random.randint(0, 2, 100).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10, 10)),
    tf.keras.layers.LSTM(8),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    run_eagerly=True,
)

print("Starting test training...")

model.fit(
    X,
    y,
    epochs=1,
    batch_size=16,
    verbose=2,
)

print("TensorFlow test completed.")