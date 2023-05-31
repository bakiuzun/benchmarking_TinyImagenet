"""
Author: Uzun Baki
"""

"""
this file is used to load a model a quantize it
float16,full int and Dynamic Range quantization is supported 
"""


import tensorflow as tf
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Inferance Test GPU')
parser.add_argument('--modele', type=str, help='modele')
parser.add_argument('--type', type=str, help='modele')
parser.add_argument('--pruning', type=bool, help='modele')
parser.add_argument('--img_size', type=int, help='modele')
args = parser.parse_args()

modele_name = args.modele
conversion_type = args.type
pruning = args.pruning
img_size = args.img_size

path = f"model/{modele_name}"
modele = None
if pruning == False:
    modele = tf.keras.models.load_model(f"{path}-model.h5")
else:
    modele = tf.keras.models.load_model(f"{path}-pruned-model.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(modele)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


def representative_data_gen():
    global modele_name
    if modele_name == "ResNet50":
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    elif modele_name == "MobileNet":
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif modele_name == "EfficientNetB1":
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

        
    data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = data_gen.flow_from_directory(
        'images_dataset/val',
        target_size=(img_size, img_size),
        batch_size=100,
        class_mode='categorical'
    )
    x,_ = next(generator)
   
    yield [x]

# PAR DÉFAUT
if conversion_type == None:
    converter.representative_dataset = representative_data_gen
    conversion_type = "DR"
    
if conversion_type == "float16":
    converter.target_spec.supported_types = [tf.float16]

if conversion_type == "full-int":
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8 


quantized_model = converter.convert()
quantized_and_tflite_file = f"model/{modele_name}-quantized-{conversion_type}-model.tflite"
tf.io.write_file(quantized_and_tflite_file, quantized_model)