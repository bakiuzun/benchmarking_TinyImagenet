"""
Author: Uzun Baki
"""

"""
file used to prune an existing model, you may need to train the model before executing this script
"""

import tensorflow
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_model_optimization.sparsity.keras import PolynomialDecay
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import UpdatePruningStep

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Inferance Test GPU')
parser.add_argument('--modele', type=str, help='modele')

args = parser.parse_args()


img_height = img_width = 120
channels = 3
path = "images_dataset"

modele_name = args.modele
modele = None 
if modele_name == "ResNet50":
    preprocess_input = tensorflow.keras.applications.resnet50.preprocess_input
elif modele_name == "MobileNet":
    preprocess_input = tensorflow.keras.applications.mobilenet.preprocess_input
elif modele_name == "EfficientNetB1":
    preprocess_input = tensorflow.keras.applications.efficientnet.preprocess_input

model_path = f"model/{modele_name}-model.h5"
modele = tensorflow.keras.models.load_model(model_path)

def get_data(batch,preprocess_input):
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



# Compute end step to finish pruning after 2 epochs.
batch_size = 16
epochs = 30

train_data,val_data  = get_data(batch_size,preprocess_input)

end_step = (train_data.n // batch_size) * epochs
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.2,
        final_sparsity=0.7,
        begin_step=0,
        end_step=end_step)

    # Prune the model
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(modele, 
                                                        pruning_schedule=pruning_schedule,
                                                        pruning_policy=tfmot.sparsity.keras.PruneForLatencyOnXNNPack())

    # Compile the pruned model
pruned_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the pruned model
pruned_model.fit(train_data,epochs=epochs,validation_data=val_data,callbacks=[UpdatePruningStep()])


model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
pruned_keras_file =  f"model/{modele_name}-pruned-model.h5"
tensorflow.keras.models.save_model(model_for_export, pruned_keras_file)