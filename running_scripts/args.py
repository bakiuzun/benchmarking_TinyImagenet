import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Inferance Test GPU')


# Add an argument to the ArgumentParser object
parser.add_argument('--batch_size', type=int, help='le batch size')
parser.add_argument('--memoire_gpu', type=int, help='memoire gpu')
parser.add_argument('--modele', type=str, help='memoire gpu')
parser.add_argument('--per_image', type=str, help='prediction per image True/False')
parser.add_argument('--cpu', type=int, help='cpu number')
parser.add_argument('--gpu_name', type=str, help='gpu_name')
parser.add_argument("--only_cpu",type=str,help="only cpu arg" )
parser.add_argument("--tflite",type=str,help="tfliteModel")
parser.add_argument("--quantization",type=str,help="quantization")
parser.add_argument("--pruning",type=str,help="pruning")
parser.add_argument("--only_eval",type=str,help="pruning")
parser.add_argument("--img_size",type=int,help="image size")


# Parse the arguments
args = parser.parse_args()
dict = {}
dict["batch_size"] =  args.batch_size
dict["memoire_gpu"] =  args.memoire_gpu
dict["modele"] =  args.modele
dict["gpu_name"] = args.gpu_name
dict["per_image"] =  args.per_image
dict["cpu"] = args.cpu
dict["only_cpu"] = args.only_cpu
dict["tflite"] = args.tflite
dict["pruning"] = args.pruning
dict["quantization"] = args.quantization
dict["only_eval"] = args.only_eval
dict["img_size"] = args.img_size
dict["path"] = None



def verify_args(dict):

    if dict["tflite"] == None:
        dict["tflite"] = "False"
    if dict["only_cpu"] == None:
       dict["only_cpu"] = "False"
    
    if dict["per_image"] == None:
       dict["per_image"] = "True"

    if dict["only_eval"] == None:
        dict["only_eval"] = "False"
    
    if dict["batch_size"] == None and dict["per_image"] == "True":
        # tiny imagenet dataset contain 10000 for validation
        dict["batch_size"] = 10000 
    
    if dict["per_image"] == "False" and dict["batch_size"] == None:
         raise RuntimeError("Batch size or Per image should be specified")


    if dict["img_size"] == None:
        raise RuntimeError("Image size can't be None")

    if dict["cpu"] == None:dict["cpu"] = 4
    if dict["memoire_gpu"] == None:dict["memoire_gpu"] = 32


    if dict["tflite"] == "True": extension = "-model.tflite"
    else: extension = "-model.h5"

    if dict["pruning"] == None:
        dict["pruning"] = "False"

    if dict["quantization"] == None:
        dict["quantization"] = "False"
    

    if dict["quantization"] == "False" and  dict["pruning"]  == "False":
        dict["path"] = "model/"+dict["modele"]+extension
    

    if   dict["quantization"]  != "False"  and dict["pruning"]  == "True":
        dict["path"] = "model/"+dict["modele"]+"-"+"pruned"+"-quantized-"+dict["quantization"]+extension
    
    if  dict["quantization"] == "False" and dict["pruning"]  == "True":
       dict["path"] = "model/"+dict["modele"]+"-pruned"+extension

    if  dict["quantization"] !=  "False" and dict["pruning"]  == "False":
        dict["path"] = "model/"+dict["modele"]+"-quantized-"+dict["quantization"]+extension
    
    
    if dict["path"] == None:
      raise RuntimeError("Path should be specified")
