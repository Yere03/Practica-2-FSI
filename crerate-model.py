import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def process_images_and_tags(directory, image_size, batch_size, porcentage,seed_value):
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=porcentage,
        subset="training",
        seed=seed_value,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=porcentage,
        subset="validation",
        seed=seed_value,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    train_ds = train_ds.prefetch(buffer_size=batch_size)
    val_ds = val_ds.prefetch(buffer_size=batch_size)
    
    return train_ds, val_ds


def create_model(image_size):
    model = keras.Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    
    return model


def training_model(model, train_ds, val_ds, epochs, patience): 

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=patience, restore_best_weights=True)

    return model.fit_generator(
            train_ds,
            epochs=epochs, 
            validation_data=val_ds,
            callbacks = [es]
            )

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


def evaluate_results(val_ds):
    results = np.concatenate([(y, model.predict(x=x)) for x, y in val_ds], axis=1)

    predictions = np.argmax(results[0], axis=1)
    labels = np.argmax(results[1], axis=1)

    cf_matrix = confusion_matrix(labels, predictions)

    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")

    print(classification_report(labels, predictions, digits = 4))


def predict_image(image):
    img = keras.preprocessing.image.load_img(image, target_size=image_size)

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    print(np.argmax(predictions[0]))


if "__main__" == __name__:
    
    seed_value = 13
    random.seed(seed_value)        
    np.random.seed(seed_value)    
    tf.random.set_seed(seed_value) 

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    
    
    image_size = (400, 400)
    batch_size = 64
    porcentage = 0.2    
    epochs=300
    patience = 40
      
      
    datasets = process_images_and_tags("fire_dataset", image_size, batch_size, porcentage, seed_value)
    
    train_ds = datasets[0]
    val_ds = datasets[1]
    
    model = create_model(image_size)
    
    trained_model = training_model(model, train_ds, val_ds, epochs, patience)
    
    
    model.save("models/2convulutionallayers20patience400x40030batchbinary3convlayersArc32-64-C.keras")
    
        
    show_plot(trained_model)
    
    evaluate_results(val_ds)



