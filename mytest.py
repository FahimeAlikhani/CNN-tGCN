import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ساختار شبکه عصبی کانولوشن
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# مشخص کردن پارامترهای شبکه کانولوشن گراف
k = 4  #  {4، 6، 8، 10، 12} 

# تعریف تابع هزینه و بهینه ساز
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# بارگیری و تقسیم مجموعه داده‌ها
folder_0 = 'archive/8863/0'
folder_1 = 'archive/8863/1'
num_images_per_class = 200
images = []
labels = []
batch_size = 32
num_epochs = 40

# تعریف لیست‌هایی برای ذخیره تصاویر تست و برچسب‌های آن‌ها
test_images = []
test_labels = []

for image_file in os.listdir(folder_0):
    image_path = os.path.join(folder_0, image_file)
    image = Image.open(image_path)
    image = image.resize((50, 50))
    image_array = np.array(image)
    images.append(image_array)
    labels.append(0)

    if len(images) == num_images_per_class:
        break

for image_file in os.listdir(folder_1):
    image_path = os.path.join(folder_1, image_file)
    image = Image.open(image_path)
    image = image.resize((50, 50))
    image_array = np.array(image)
    images.append(image_array)
    labels.append(1)

    if len(images) == num_images_per_class * 2:
        break

images = np.array(images)
labels = np.array(labels)

# تقسیم داده‌ها به دسته‌های آموزش، تست و اعتبارسنجی
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, random_state=42
)
test_images, val_images, test_labels, val_labels = train_test_split(
    test_images, test_labels, test_size=0.66, random_state=42
)

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
val_labels = tf.keras.utils.to_categorical(val_labels)

# آموزش شبکه با داده‌های آموزشی
model.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs, validation_data=(val_images, val_labels))

# استفاده از مدل برای استخراج داده‌های ویژگی تصاویر تست
test_features = model.predict(test_images)

# جایگزینی feature_data با داده‌های ویژگی تصاویر تست
feature_data = test_features
# ساخت گراف k-NN بر اساس داده‌های ویژگی
graph = NearestNeighbors(n_neighbors=k)
graph.fit(feature_data)

# استفاده از مدل برای دسته‌بندی تصاویر تست
test_predictions = model.predict(test_images)

# دسته‌بندی تصاویر تست با استفاده از گراف k-NN
for i in range(len(test_images)):
    distances, indices = graph.kneighbors(test_predictions[i].reshape(1, -1))
    nearest_labels = labels[indices]
    # انجام عملیات مورد نیاز بر روی برچسب‌های نزدیکترین همسایه‌ها

# ارزیابی شبکه با داده‌های تست
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

##################### رسم نمودار
# آموزش شبکه با داده‌های آموزشی و ذخیره تاریخچه آموزش
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs, validation_data=(val_images, val_labels))

# دسترسی به مقادیر دقت و خطا از تاریخچه آموزش
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# رسم نمودارها در یک شبکه
plt.figure(figsize=(12, 6))

# نمودار دقت
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# نمودار خطا
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# نمایش نمودارها
plt.tight_layout()
plt.show()