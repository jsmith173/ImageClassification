import matplotlib.pyplot as plt
import numpy as np
import PIL, json, glob
import tensorflow as tf
import pathlib

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


with open('config.json', 'r') as f:
 config = json.load(f)

# ## Download and explore the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')


# After downloading, you should now have a copy of the dataset available. There are 3,670 total images:
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# ## Load data using a Keras utility
batch_size = 32
img_height = 180
img_width = 180


# It's good practice to use a validation split when developing your model. Use 80% of the images for training and 20% for validation.
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
class_names = train_ds.class_names
print(class_names)


# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Or, you can include the layer inside your model definition, which can simplify deployment. Use the second approach here.
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

