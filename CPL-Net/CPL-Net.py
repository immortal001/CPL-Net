import tensorflow as tf
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from keras import backend as K

# 定义数据集路径
train_data_dir = ''
test_data_dir = ''

# 定义模型
input_shape = (64, 64, 3)

cnn_model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten()
])

lstm_model = tf.keras.Sequential([
    layers.Reshape((64, -1)),
    layers.LSTM(192)
])


input_data = layers.Input(shape=input_shape)
cnn_output = cnn_model(input_data)  # 使用 CNN 模型提取特征
lstm_output = lstm_model(cnn_output)  # 使用 LSTM 模型提取特征


concatenated_output = tf.concat([cnn_output, lstm_output], axis=1)


num_classes = len(os.listdir(train_data_dir))
output = layers.Dense(num_classes, activation='softmax')(concatenated_output)


model = tf.keras.Model(inputs=input_data, outputs=output)

def macro_f1(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy',macro_f1])


batch_size = 8
train_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(
    train_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical')
test_image_generator = ImageDataGenerator(rescale=1./255)
test_data_gen = test_image_generator.flow_from_directory(
    test_data_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical')


epochs = 50
history = model.fit(train_data_gen, epochs=epochs, validation_data=test_data_gen)


config = {
    "font.family":'Times New Roman',
    "axes.unicode_minus": False,
}
rcParams.update(config)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CPL-Net accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])


plt.subplot(1, 2, 2)
plt.plot(history.history['macro_f1'], label='Train Macro F1 Score')
plt.plot(history.history['val_macro_f1'], label='Validation Macro F1 Score')
plt.title('Macro F1 Score')
plt.ylabel('Macro F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])

plt.tight_layout()
plt.show()