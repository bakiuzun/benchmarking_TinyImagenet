"""
Authors: Uzun Baki
"""

"""
file used to fine-tune a pre-trained model and save it 
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import keras
import tensorflow
import argparse


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--modele', type=str, help='modele name ')
parser.add_argument('--img_size', type=int, help='image size')

args = parser.parse_args()

modele_name = args.modele
img_size = args.img_size

path = "images_dataset"
img_height = img_width = img_size
channels = 3


preprocess_input = None
base = None

if modele_name == "ResNet50":
    preprocess_input = tensorflow.keras.applications.resnet50.preprocess_input
    base =  tensorflow.keras.applications.ResNet50(
                                include_top=False,
                                weights="imagenet",
                                input_shape=(img_width,img_height,channels))
        

elif modele_name == "MobileNet":
    preprocess_input = tensorflow.keras.applications.mobilenet.preprocess_input
    base =  tensorflow.keras.applications.MobileNet(
                            include_top=False,
                            weights="imagenet",
                            input_shape=(img_width,img_height,channels))
    
elif modele_name == "EfficientNetB1":
    preprocess_input = tensorflow.keras.applications.efficientnet.preprocess_input
    base =  tensorflow.keras.applications.EfficientNetB1(
                                include_top=False,
                                weights="imagenet",
                                input_shape=(img_width,img_height,channels))



# Set up the data generator
def get_data(batch):
    global preprocess_input

    data_gen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2)
        
# Load the data from the directory
    train_data = data_gen.flow_from_directory(
        path+"/train",
        target_size=(img_width, img_height),
        batch_size=batch,
        class_mode='categorical',
    )

    val_data = data_gen.flow_from_directory(
        path+"/val",
        target_size=(img_width, img_height),
        batch_size=batch,
        class_mode='categorical',
    )
    return train_data,val_data



def get_model(base):

    base.trainable = False

    model = models.Sequential()
    model.add(base)
    model.add(layers.GlobalAveragePooling2D())

    ## the tiny imagenet dataset contain 200 classes
    model.add(layers.Dense(200, activation='softmax'))

    return model


callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'model/{modele_name}-model.h5',
            monitor='val_loss',
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.01, 
            patience=5, 
            min_lr=0.0001
        )
    ]


learning_rate = 0.01
optimizer = tensorflow.keras.optimizers.Adam(learning_rate)

model = get_model(base)
train_data,val_data = get_data(32)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data,epochs=80,validation_data=val_data,callbacks=callbacks_list)


