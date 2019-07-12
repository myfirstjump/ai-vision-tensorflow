from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataGenerator(object):
    def __init__(self):
        pass

    def tf_image_generator(self, train_dir, validation_dir):
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(300, 300),
            batch_size=128,
            class_mode='binary'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(300, 300),
            batch_size=128,
            class_mode='binary'
        )