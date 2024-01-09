import keras_preprocessing.image
from keras.models import load_model

import tensorflow as tf


images = {"1.jpg": 1, "2.png": 0, "3.jpg": 1, "4.webp": 1, "5.webp": 1, "6.jpg": 1, "7.jpg": 1, "8.jpg": 0, "9.jpg": 0,
          "10.jpg": 0,"11.jpeg": 0,"12.jpg": 0,"13.jpg": 0,"14.jpg": 1,"15.jpg": 1}


def model_results(image_size, model_path):
    model = load_model(model_path)
    results = 0

    for key in images:
        image_path = "test_images\\" + key
        img = keras_preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = keras_preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        prediction = model.predict(img_array)
        if prediction[0] > 0.5:
            prediction = 1
        else:
            prediction = 0

        if prediction == images[key]:
            print("Acierto" + key)
            results += 1

    return results


print("model1: " , model_results((400, 400), "models/2convulutionallayers20patience400x40030batchbinary3convlayersArc32-64-C.keras"))
print("model2: " , model_results((400, 400), "models/2convulutionallayers20patience400x40030batchbinary3convlayersArc32-64-Z.keras"))

