from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
from keras import backend as K


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    '',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    '',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)


model = models.Sequential()
model.add(layers.Reshape((64, 64), input_shape=(64, 64, 1)))
model.add(layers.LSTM(64))
model.add(layers.Dense(25, activation='softmax'))




def macro_f1(y_true, y_pred):


    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


model.compile(optimizer='adam',
              loss='categorical_crossentropy', # 更改为categorical_crossentropy
              metrics=['accuracy',macro_f1])


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)


test_metrics = model.evaluate(test_generator, steps=len(test_generator), verbose=2)
test_loss, test_acc, test_f1_macro = test_metrics
print('\nTest accuracy:', test_acc)
print('Test Macro F1 Score:', test_f1_macro)


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['macro_f1'], label='Train Macro F1 Score')
plt.plot(history.history['val_macro_f1'], label='Validation Macro F1 Score')
plt.title('Macro F1')
plt.ylabel('Macro F1 Score')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()