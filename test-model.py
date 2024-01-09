import keras_preprocessing.image
from keras.models import load_model
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_plot(model):
    import matplotlib.pyplot as plt
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.plot(model.history['loss'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation','loss'], loc='upper right')
    plt.show()



image_path="test_images/15.jpg"
image_size =(400,400)
model = load_model('models/2convulutionallayers20patience400x40030batchbinary3convlayersArc32-64-Z.keras')


img = keras_preprocessing.image.load_img(image_path, target_size=image_size)

img_array = keras_preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
print(predictions)
print(predictions[0])
#print("No hay fuego" if np.argmax(predictions[0]) else "Hay fuego")
print("No hay fuego" if 0.5 < predictions[0] else "Hay fuego")

imagen = mpimg.imread(image_path)

plt.imshow(imagen)
plt.axis('off') 
plt.show()