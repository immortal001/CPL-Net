import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from matplotlib import rcParams
from keras import backend as K
# 定义模型
model = tf.keras.Sequential([
    # 第一个卷积层
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # 第二个卷积层
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 第三个卷积层
    layers.Conv2D(64, (3, 3), activation='relu'),

    # 全连接层
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(25,activation='softmax')
])

# 加载自己的数据集
train_dir = ''
test_dir = ''

# 使用ImageDataGenerator加载图像数据（无数据增强）
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

def macro_f1(y_true, y_pred):
    # 将预测值转换为one-hot编码的形式

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy',macro_f1])

loss='categorical_crossentropy'
# 定义训练过程中的回调函数
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        print(f'Epoch {epoch + 1}: Training loss = {logs.get("loss")}, Training accuracy = {logs.get("accuracy")}')
        print(
            f'Epoch {epoch + 1}: Validation loss = {logs.get("val_loss")}, Validation accuracy = {logs.get("val_accuracy")}')
        print('--------------------------------------------')


# 创建回调对象
callback = MyCallback()

# 训练模型
history = model.fit(train_generator, epochs=50, validation_data=validation_generator, callbacks=[callback])

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False,#解决负号无法显示的问题
}
rcParams.update(config)
# 绘制准确率曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['macro_f1'], label='Train Macro F1 Score')
plt.plot(history.history['val_macro_f1'], label='Validation Macro F1 Score')
plt.title('Macro F1 Score')
plt.ylabel('Macro F1 Score')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()