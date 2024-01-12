import matplotlib.pyplot as plt
import numpy as np
import PIL, json, glob
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing import image

def pprint(a):
 arr = a.numpy()
 print(class_names)
 s = 'All items distribution = ['
 for i in range(len(arr)):
  s = s+str("%0.2f" % (100*arr[i]))+'% '
 s = s+']' 
 print(s) 

def load_image(fn):
 img = tf.keras.utils.load_img(
    fn, target_size=(img_height, img_width)
 )
 img_array = tf.keras.utils.img_to_array(img)
 img_array = tf.expand_dims(img_array, 0) # Create a batch
 return img_array

def run_predict(fn, img_array):
 predictions = model.predict(img_array)
 score = tf.nn.softmax(predictions[0])
 shape = score.shape
 #print(f"score.shape: {shape}")
 pprint(score)
 print(
    "{} most likely belongs to {} with a {:.2f} percent confidence."
    .format(fn, class_names[np.argmax(score)].upper(), 100 * np.max(score))
 )


sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# ## Download and explore the dataset
with open('config.json', 'r') as f:
 config = json.load(f)

import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')


# After downloading, you should now have a copy of the dataset available. There are 3,670 total images:
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# Here are some roses:

# In[5]:


roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))


# In[6]:


PIL.Image.open(str(roses[1]))


# And some tulips:

# In[7]:


tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))


# In[8]:


PIL.Image.open(str(tulips[1]))


# ## Load data using a Keras utility
# 
# Next, load these images off disk using the helpful `tf.keras.utils.image_dataset_from_directory` utility. This will take you from a directory of images on disk to a `tf.data.Dataset` in just a couple lines of code. If you like, you can also write your own data loading code from scratch by visiting the [Load and preprocess images](../load_data/images.ipynb) tutorial.

# ### Create a dataset

# Define some parameters for the loader:

# In[9]:


batch_size = 32
img_height = 180
img_width = 180


# It's good practice to use a validation split when developing your model. Use 80% of the images for training and 20% for validation.

# In[10]:


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[11]:


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# You can find the class names in the `class_names` attribute on these datasets. These correspond to the directory names in alphabetical order.

# In[12]:


class_names = train_ds.class_names
print(class_names)


# ## Visualize the data
# 
# Here are the first nine images from the training dataset:

# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# You will pass these datasets to the Keras `Model.fit` method for training later in this tutorial. If you like, you can also manually iterate over the dataset and retrieve batches of images:

# In[14]:


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.
# 
# You can call `.numpy()` on the `image_batch` and `labels_batch` tensors to convert them to a `numpy.ndarray`.
# 

# ## Configure the dataset for performance
# 
# Make sure to use buffered prefetching, so you can yield data from disk without having I/O become blocking. These are two important methods you should use when loading data:
# 
# - `Dataset.cache` keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
# - `Dataset.prefetch` overlaps data preprocessing and model execution while training.
# 
# Interested readers can learn more about both methods, as well as how to cache data to disk in the *Prefetching* section of the [Better performance with the tf.data API](../../guide/data_performance.ipynb) guide.

# In[15]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ## Standardize the data

# The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network; in general you should seek to make your input values small.
# 
# Here, you will standardize values to be in the `[0, 1]` range by using `tf.keras.layers.Rescaling`:

# In[16]:


normalization_layer = layers.Rescaling(1./255)


# There are two ways to use this layer. You can apply it to the dataset by calling `Dataset.map`:

# In[17]:


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


# Or, you can include the layer inside your model definition, which can simplify deployment. Use the second approach here.

# Note: You previously resized images using the `image_size` argument of `tf.keras.utils.image_dataset_from_directory`. If you want to include the resizing logic in your model as well, you can use the `tf.keras.layers.Resizing` layer.

# ## A basic Keras model
# 
# ### Create the model
# 
# The Keras [Sequential](https://www.tensorflow.org/guide/keras/sequential_model) model consists of three convolution blocks (`tf.keras.layers.Conv2D`) with a max pooling layer (`tf.keras.layers.MaxPooling2D`) in each of them. There's a fully-connected layer (`tf.keras.layers.Dense`) with 128 units on top of it that is activated by a ReLU activation function (`'relu'`). This model has not been tuned for high accuracy; the goal of this tutorial is to show a standard approach.

# In[18]:


num_classes = len(class_names)

if config['gen_model'] == 1:
 model = Sequential([
   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
   layers.Conv2D(16, 3, padding='same', activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(32, 3, padding='same', activation='relu'),
   layers.MaxPooling2D(),
   layers.Conv2D(64, 3, padding='same', activation='relu'),
   layers.MaxPooling2D(),
   layers.Flatten(),
   layers.Dense(128, activation='relu'),
   layers.Dense(num_classes)
 ])

 # ### Compile the model
 model.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])


 # ### Model summary
 model.summary()

 # ### Train the model
 epochs=10
 history = model.fit(
   train_ds,
   validation_data=val_ds,
   epochs=epochs
 )
 model.save('model.keras')
 converter = tf.lite.TFLiteConverter.from_keras_model(model)
 tflite_model = converter.convert()
 with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

 # ## Visualize training results
 acc = history.history['accuracy']
 val_acc = history.history['val_accuracy']

 loss = history.history['loss']
 val_loss = history.history['val_loss']

 epochs_range = range(epochs)
else:
 model = tf.keras.models.load_model('model.keras')

#Predict
input_files = glob.glob("*.jpg")
print(input_files)
print('')
print('>> Starting predictions...')

for img_file in input_files:
 img1 = load_image(img_file)
 run_predict(img_file.upper(), img1)
 print('')

