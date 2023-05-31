import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# Load the Keras model (.h5 file)
model = load_model('model/MobileNet-model.h5')
print('TensorFlow:', tf.__version__)


forward_pass = tf.function(
    model.call,
    input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
graph_info = profile(forward_pass.get_concrete_function().graph,
                        options=ProfileOptionBuilder.float_operation())

# The //2 is necessary since `profile` counts multiply and accumulate
# as two flops, here we report the total number of multiply accumulate ops
flops = graph_info.total_float_ops // 2
print('Flops: {:,}'.format(flops))
print("Flops with counting",flops*2)
print('Flops M', flops / 1.0e6)