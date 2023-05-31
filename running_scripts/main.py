"""
Authors: Uzun Baki
"""

"""
this file is used to make a benchmark on a model by making prediction on the validation set and storing useful informations
such as GPU utilization,CPU utilization and more at the same time.
refers to the args.py file to understand arguments 

"""

from imports import * 

if args.dict['only_cpu'] == 'True':
    tf.config.set_visible_devices([], 'GPU')


stop_recording = False

img_width = img_height = args.dict["img_size"]

def get_model():
    model = None
    if args.dict['tflite'] == 'True':
        model = tf.lite.Interpreter(args.dict["path"])
        model.allocate_tensors()
    else:
        model = tf.keras.models.load_model(args.dict["path"])
    return model

def load_data(batch_size):
    dataset_path = "images_dataset/val"
    preprocess_input = None

    if args.dict['modele'] == "ResNet50":
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    elif args.dict['modele'] == "MobileNet":
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif args.dict['modele'] == "EfficientNetB0":
         preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    dtype = "float32"
    if args.dict['quantization'] == "full-int":
        dtype = "uint8"

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,dtype=dtype)
    
    # Load the data from the directory
    train_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
    )
    return train_data

def tflite_predict(model,data):
    base = time.time()
    input_index = model.get_input_details()[0]["index"]
    for _ in range(data.n // args.dict["batch_size"]):
        images,_  = next(data)
        model.set_tensor(input_index,images)
        model.invoke()
 
def predict(model,data):
    for _ in range(data.n // args.dict["batch_size"]):
        images,_ = next(data)
        pred = model(images)

 
def evaluate_model(model,data):
    
    acc = 0
    if args.dict['tflite'] == 'True':
        input_index = model.get_input_details()[0]["index"]
        output_tensor = model.get_output_details()[0]["index"]
        for _ in range(data.n // args.dict['batch_size']):
            images,true_classes = next(data)
            model.set_tensor(input_index, images)
            model.invoke()
            preds = model.get_tensor(output_tensor)
            if np.argmax(preds) == np.argmax(true_classes):
                acc += 1
    else:
        _, acc = model.evaluate(data)

    with open(f"{args.dict['modele']}_evaluation.txt","a") as f:
        f.write(f"Evaluation modele {args.dict['modele']}, Quantized {args.dict['quantization']}, Pruned {args.dict['pruning']},Tflite {args.dict['tflite']}\n")
        f.write(f"Test accuracy: {acc}\n")
        f.write(f"Number Images: {data.n}\n")


def store_info(base_time):
    global stop_recording

    header = False
    header_str = "utilization_gpu,memory_used,memory_total,power_draw,cpu_usage,time,"
    file_name = f"modele_{args.dict['modele']}_quantized_{args.dict['quantization']}_pruned_{args.dict['pruning']}_tflite_{args.dict['tflite']}_batch_{args.dict['batch_size']}_memoire-gpu_{args.dict['memoire_gpu']}_per-image_{args.dict['per_image']}_cpu-number_{args.dict['cpu']}_gpu-name_{args.dict['gpu_name']}_only-cpu_{args.dict['only_cpu']}_.csv"

    while True:
        if stop_recording == True:break
        cmd = ['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader']

        # Execute the command and capture the output
        result = subprocess.run(cmd, stdout=subprocess.PIPE)

        output = result.stdout.decode('utf-8')
        x = output.split(",")
        x[0] = "\n"+x[0]
        x[-1] = x[-1].strip()
        x.append(f"{psutil.cpu_percent()}")
        x.append(f"{round(time.time() - base_time,2)},")
        # Write the output to a file
        with open(file_name, 'a') as f:
            val = ','.join(x)
            if header == False:
                val = header_str
                header = True

            f.write(val)
        
        time.sleep(1)



def run():
    global stop_recording
    args.verify_args(args.dict)

    
    data = load_data(args.dict['batch_size'])
    modele = get_model()

    if args.dict["only_eval"] == 'False':

        gpu_thread = threading.Thread(target=store_info, args=(time.time(),))
        gpu_thread.start()

        if args.dict["tflite"] == 'True':
            tflite_predict(modele,data)
        else:
            predict(modele,data)

        stop_recording = True
        gpu_thread.join()


    evaluate_model(modele,data)
    

if __name__ == "__main__": run()
