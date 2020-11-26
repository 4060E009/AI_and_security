# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

## Import載入 the Fashion MNIST dataset

###直接從TensorFlow導入和加載Fashion MNIST數據

fashion_mnist = tf.keras.datasets.fashion_mnist

###加載數據集將返回四個NumPy數組

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

##Loading the dataset returns four NumPy arrays:

###每個圖像都映射到一個標籤。由於類名稱不包含在數據集中，因此將它們存儲在此處以供以後在繪製圖像時使用

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## Explore the data

###探索數據，在訓練模型之前，讓我們探索數據集的格式

train_images.shape

len(train_labels)

train_labels

test_images.shape

len(test_labels)

## Preprocess the data

###預處理數據，在訓練網絡之前，必須對數據進行預處理

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

## Scale these values to a range of 0 to 1 before feeding them to the neural network model. 

###將這些值縮放到0到1的範圍，然後再將其輸入神經網絡模型。為此，將值除以255。以相同的方式預處理訓練集和測試集

train_images = train_images / 255.0

test_images = test_images / 255.0

## To verify that the data is in the correct format 

###為了驗證數據的格式正確，並準備好構建和訓練網絡，讓我們顯示訓練集中的前25個圖像，並在每個圖像下方顯示類別名稱

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

### 建立模型 Build the model
### KERAS開發模式: 1.Sequential(堆積木)  2.FUCTIONAL api   3.SUBCLASS(物件導向  子類別)   

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

### 顯示模型==>使用model.summary()函數

model.summary()

### 設定模型訓練所需的參數==>使用model.compile()函數

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

### 訓練模型Train the model==>使用model.fit()函數

model.fit(train_images, train_labels, epochs=10)

### 計算模型預測的正確性Evaluate accuracy==>使用model.evaluate()函數

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

### 進行預測Make predictions==>使用predict()函數

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

test_labels[0]

### 其他

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
                                
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

### Verify predictions

###第0張圖片，預測和預測數組。正確的預測標籤為藍色，錯誤的預測標籤為紅色。該數字給出了預測標籤的百分比（滿分為100）

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

###繪製一些帶有預測的圖像。請注意，即使非常自信，該模型也可能是錯誤的

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

## Use the trained model

###使用經過訓練的模型對單個圖像進行預測

# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

###為該圖像預測正確的標籤

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

###tf.keras.Model.predict返回列表列表-數據批次中每個圖像的一個列表。批量獲取我們（僅）圖像的預測

np.argmax(predictions_single[0])
