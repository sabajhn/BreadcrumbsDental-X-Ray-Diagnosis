
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

def get_train_data_gen():
    return ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='reflect'
    )

def get_val_data_gen():
    return ImageDataGenerator(rescale=1.0/255.0)

def load_data(data_gen, directory):
    return data_gen.flow_from_directory(
        directory,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )