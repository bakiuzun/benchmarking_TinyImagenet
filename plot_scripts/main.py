"""
Author: Uzun Baki
"""

# Rest of the code...


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import os 




def return_dict(file):
    dict = {}
    for i in range(0,len(file)-1,2):
        dict[file[i]] = file[i+1]
    return dict

def plot_models(modeles):
    """
    Plot performance metrics for a list of models.
    you'll be able to see flops,acc, and img pred per second comparisons
    on 3 axis,
    in this example we will have comparisons between ResNet50 and MobileNet
    it will be ordered by the flops for each model. The flops calculated 
    for quantized model are the same as the base model.
    
    Arguments:
    modeles -- A list of model names (strings).
    """
    # image sizes used in the benchmark
    sizes = [120,224,512]

    # the path that stores the files for each image sizes and each model
    path = f"csv/tiny_imagenet/titan/"
    files = sorted([f for f in os.listdir(path+f"{sizes[0]}/{modeles[0]}")])

    ## we divided data from CPU and from GPU 
    names_cpu = []
    names_gpu = []
    accs_cpu = []
    accs_gpu = []

    flops_cpu = []
    flops_gpu = []
    img_per_Sec_cpu = []
    img_per_Sec_gpu = []
    accs = []

    numbers_of_images = 10000

    for i in range(len(files)):
        base, _ = os.path.splitext(files[i])
        splitted_file = base.split("_")
       
        dataframes = []

        ## each dicts contains useful informations about the data that you can explore by printing the dictionnary
        dicts = []
        for x in sizes:
            for j in range(len(modeles)):
                splitted_file[1] = modeles[j]
                new_path =  '_'.join(splitted_file) + ".csv"

                dataframes.append(pd.read_csv(path+f"{x}/{modeles[j]}/"+new_path))
                the_dict = return_dict(splitted_file)
                the_dict["img_size"] = x
                dicts.append(the_dict)


        for i in range(len(dicts)):
            if dicts[i]["quantized"] == "full-int":
                names_cpu.append(f"{name} FL {img_size} ")
            
            name = dicts[i]['modele']
            img_size = str(dicts[i]['img_size'])[0]

            if dicts[i]["only-cpu"] == "True":
                if dicts[i]['quantized'] == "float32":
                    names_cpu.append(f"{name} DR {img_size} ")
                elif dicts[i]["quantized"] == "float16":
                    names_cpu.append(f"{name} f16 {img_size}")
                else:
                    names_cpu.append(f"{name} CPU {img_size} ")
                img_per_Sec_cpu.append(numbers_of_images// dataframes[i].time.max())
                accs_cpu.append(dataframes[i].acc.max() * 100)
                flops_cpu.append(dataframes[i].flops.max() /  1e9)
            else:
                img_per_Sec_gpu.append(numbers_of_images// dataframes[i].time.max())
                names_gpu.append(f"{name} GPU {img_size} ")
                accs_gpu.append(dataframes[i].acc.max() * 100)
                flops_gpu.append(dataframes[i].flops.max() /  1e9)


            accs.append(dataframes[i].acc.max() * 100)

    sorted_indices = np.argsort(np.concatenate((flops_cpu, flops_gpu), axis=0))
    sorted_names = np.concatenate((names_cpu, names_gpu), axis=0)[sorted_indices]
    sorted_flops = np.concatenate((flops_cpu, flops_gpu), axis=0)[sorted_indices]
    sorted_acc = np.concatenate((accs_cpu, accs_gpu), axis=0)[sorted_indices]
    sorted_img_per_sec = np.concatenate((img_per_Sec_cpu, img_per_Sec_gpu), axis=0)[sorted_indices]
    
    cpu_indices = [i for i, name in enumerate(sorted_names) if 'GPU' not in name]
    gpu_indices = [i for i, name in enumerate(sorted_names) if 'GPU' in name]

    
    fig, ax1 = plt.subplots()
    print(sorted_names)
    ax1.bar(cpu_indices, sorted_img_per_sec[cpu_indices], label='Img pred/s CPU')
    ax1.bar(gpu_indices, sorted_img_per_sec[gpu_indices], label='Img pred/s GPU')
    ax1.tick_params(axis='x',rotation=45, labelsize=8)
    ax1.grid(False)

    ax1.set_ylabel('Img Pred/s')
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(sorted_names, sorted_acc, 'r-',marker="o" ,label='Accuracy %')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='x', rotation=45,labelsize=8)
    ax2.set_ylim(0, 100)

    ax3 = ax1.twinx()
    # Plot line chart on axis #3
    ax3.spines['right'].set_position(('outward', 50))
    ax3.plot(sorted_names, sorted_flops,color="purple" ,marker="o" ,label='GFLOPS')
    ax3.grid(False) # turn off grid #3
    ax3.set_ylabel('GFLOPS')
    ax3.tick_params(axis='x',rotation=45,labelsize=8)
    ax3.yaxis.set_major_formatter('{:.1f}'.format)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3,labels1 +  labels2 + labels3 ,loc='upper right', bbox_to_anchor=(1, 1.15))
    plt.title(f'{" ".join(modeles)}' )
    plt.show()


plot_models(["ResNet50","MobileNet"])









"""

the code below was used for another benchmarking that i did on Aerial Scene Classification Dataset,
it can be useful to use it if you want to compare the performance on different GPU. to make it run you may need to change some path as i stopped to update the code
dataset: https://captain-whu.github.io/AID/
"""



def le_max_min(titan_att,rtx_att,a6000_att,words,is_min=False,prediction=False):
    ret = words 
    if is_min == False:
        maxi =  max([titan_att.max(),rtx_att.max(),a6000_att.max()])
        if maxi == titan_att.max():
            ret += "Titan "
        if maxi == rtx_att.max():
            ret += " RTX 2080"
        if maxi == a6000_att.max():
            ret += " a6000"
    else:
        if prediction:
            maxi =  min([titan_att.max(),rtx_att.max(),a6000_att.max()])
            if maxi == titan_att.max():
                ret += "Titan "
            if maxi == rtx_att.max():
                ret += " RTX 2080 " 
            if maxi == a6000_att.max():
                ret += " a6000"
        else:
            maxi =  min([titan_att.min(),rtx_att.min(),a6000_att.min()])
            if maxi == titan_att.min():
                ret += "Titan "
            if maxi == rtx_att.min():
                ret += " RTX 2080 " 
            if maxi == a6000_att.min():
                ret += " a6000"

    return ret+f" = {maxi} \n"


def write_info(gpu_name,pandas_DataFrame):
    global_info = f"GPU: {gpu_name} Preds:{pandas_DataFrame.time.max()}s "
    gpu_info_percentage = f"Max GPU: {pandas_DataFrame.utilization_gpu.max()}%,Mean:{round(pandas_DataFrame.utilization_gpu.mean(),2)}%,Min:{round(pandas_DataFrame.utilization_gpu.min(),2)} "
    gpu_info_watt = f"Max GPU Watt: {pandas_DataFrame.poer_dra.max()}, Mean GPU Watt: {round(pandas_DataFrame.poer_dra.mean(),2)}, Min GPU Watt: {round(pandas_DataFrame.poer_dra.min(),2)} "
    cpu_info =  f"Max CPU: {pandas_DataFrame.cpu_usage.max()}, Mean CPU: {round(pandas_DataFrame.cpu_usage.mean(),2)},Min CPU:{round(pandas_DataFrame.cpu_usage.min(),2)} "
    memory_info = f"Memory Total: {pandas_DataFrame.memory_total.max()}"
    ret = global_info + gpu_info_percentage + gpu_info_watt +cpu_info +memory_info

    return  ret+"\n"

def get_title_from_dict(dict):
    return f"Modele {dict['modele']},GPU 1,CPU number: 4, Memoire GPU: {dict['memoire-gpu']}G,Only cpu: {dict['only-cpu']}, Per-Image: {dict['per-image']},Batch size {dict['batch']},Quantized {dict['quantized']},Pruned {dict['pruned']},\n" 

def write_file(filename,titan,a6000,rtx2080,dict):
    with open(filename,"a") as f:
       
        max_gpu_use =  le_max_min(titan.utilization_gpu,rtx2080.utilization_gpu,a6000.utilization_gpu,"Max utilisation du gpu en % atteint par ")
        min_gpu_use =  le_max_min(titan.utilization_gpu,rtx2080.utilization_gpu,a6000.utilization_gpu,"Min utilisation du gpu en % atteint par ",is_min=True)
        
        max_gpu_watt_use = le_max_min(titan.poer_dra,rtx2080.poer_dra,a6000.poer_dra,"Max utilisation du gpu en Watt atteint par ")
        min_gpu_watt_use = le_max_min(titan.poer_dra,rtx2080.poer_dra,a6000.poer_dra,"Min utilisation du gpu en Watt atteint par ",is_min=True)
        
        max_cpu_use = le_max_min(titan.cpu_usage,rtx2080.cpu_usage,a6000.cpu_usage,"Max utilisation du cpu atteint par ")
        min_cpu_use = le_max_min(titan.cpu_usage,rtx2080.cpu_usage,a6000.cpu_usage,"Min utilisation du cpu atteint par ",is_min=True)

        le_max_time = le_max_min(titan.time,rtx2080.time,a6000.time,"Max Temps de prédiction atteint par ")
        le_min_time = le_max_min(titan.time,rtx2080.time,a6000.time,"Min Temps de prédiction atteint par ",is_min=True,prediction=True)
        
        infos = [get_title_from_dict(dict),max_gpu_use,min_gpu_use,max_gpu_watt_use,min_gpu_watt_use,max_cpu_use,min_cpu_use,le_max_time,le_min_time]
        for i in infos:f.write(i)
        
        table = [titan,a6000,rtx2080]
        names= ["Titan","a6000","rtx2080"]

        for i,name in zip(table,names):
            f.write(write_info(name,i))

        f.write("\n\n")


def write_best_performances_preds(dir_non_tflite,non_tflite_file,tflite_file,dir_tflite,modele_name):

    tflite_dict = return_dict( tflite_file.split("_"))
    non_tflite_dict = return_dict(non_tflite_file.split("_")
                                  )
    tflite_title = get_title_from_dict(tflite_dict)
    non_tflite_title = get_title_from_dict(non_tflite_dict)

    df_tflite = pd.read_csv(dir_tflite+"/"+tflite_file)
    df_non_tflite = pd.read_csv(dir_non_tflite+"/"+non_tflite_file)

    with open(f"{modele_name}_best_performance_preds.txt","a") as f:
        f.write(f"Meilleur Performances de prédiction pour le modèle {modele_name}\n")
        f.write("TFLite Modele Quantifier \n")
        f.write(tflite_title)
        f.write(write_info(tflite_dict["gpu-name"],df_tflite)+"\n")
        f.write("Modele Base (non tflite) \n")
        f.write(non_tflite_title)
        f.write(write_info(non_tflite_dict["gpu-name"],df_non_tflite)+"\n")





def write_performances():
    min_pred_non_tflite = 1000
    min_pred_non_tflite_file = ""
    min_pred_tflite = 1000
    min_pred_tflite_file = ""
    dir_non_tflite = ""
    dir_tflite = ""


    modele_name = "vgg16"
    path = f"{modele_name}_one_batch_comparaison.txt"
    with open(path, "a") as f:f.write((f"Comparaison, Titan RTX 2080 a6000 Pour le modèle {modele_name} \n"))

    directory_2080 = f"csv/one_batch_comparaison/2080/{modele_name}"
    directory_a6000 = f"csv/one_batch_comparaison/a6000/{modele_name}"
    directory_titan = f"csv/one_batch_comparaison/titan/{modele_name}"

    files_2080 = [f for f in os.listdir(directory_2080) if os.path.isfile(os.path.join(directory_2080, f))]
    files_a6000 = [f for f in os.listdir(directory_a6000) if os.path.isfile(os.path.join(directory_a6000, f))]
    files_titan = [f for f in os.listdir(directory_titan) if os.path.isfile(os.path.join(directory_titan, f))]



    for file_2080,file_a6000,file_titan in zip(sorted(files_2080),sorted(files_a6000),sorted(files_titan)):
        base, ext = os.path.splitext(file_a6000)
        splitted_file = file_a6000.split("_")
        dict = return_dict(splitted_file)

        df_file_a6000 = pd.read_csv(directory_a6000+"/"+file_a6000)
        df_file_titan = pd.read_csv(directory_titan+"/"+file_titan)
        df_file_2080 = pd.read_csv(directory_2080+"/"+file_2080)

        le_min_time = min([df_file_titan.time.max(),df_file_2080.time.max(),df_file_a6000.time.max()])

        if dict["tflite"] == "False":
            if (min_pred_non_tflite > le_min_time):
                min_pred_non_tflite = le_min_time
                if le_min_time == df_file_titan.time.max():
                    min_pred_non_tflite_file = file_titan
                    dir_non_tflite = directory_titan
                elif le_min_time == df_file_2080.time.max():
                    min_pred_non_tflite_file = file_2080
                    dir_non_tflite = directory_2080
                else:
                    min_pred_non_tflite_file = file_a6000
                    dir_non_tflite = directory_a6000
        else:
            if (min_pred_tflite > le_min_time):
                min_pred_tflite = le_min_time
                if le_min_time == df_file_titan.time.max():
                    min_pred_tflite_file = file_titan
                    dir_tflite = directory_titan
                elif le_min_time == df_file_2080.time.max():
                    min_pred_tflite_file = file_2080
                    dir_tflite = directory_2080
                else:
                    min_pred_tflite_file = file_a6000
                    dir_tflite = directory_a6000
        write_file(path,df_file_titan,df_file_a6000,df_file_2080,dict)

    write_best_performances_preds(dir_non_tflite,min_pred_non_tflite_file,min_pred_tflite_file,dir_tflite,modele_name)




import matplotlib.ticker as ticker


def plot_performances():

    modele_name = "vgg16"
    directory_2080 = f"csv/one_batch_comparaison/2080/{modele_name}"
    directory_a6000 = f"csv/one_batch_comparaison/a6000/{modele_name}"
    directory_titan = f"csv/one_batch_comparaison/titan/{modele_name}"

    files_2080 = [f for f in os.listdir(directory_2080) if os.path.isfile(os.path.join(directory_2080, f))]
    files_a6000 = [f for f in os.listdir(directory_a6000) if os.path.isfile(os.path.join(directory_a6000, f))]
    files_titan = [f for f in os.listdir(directory_titan) if os.path.isfile(os.path.join(directory_titan, f))]


    for file_2080,file_a6000,file_titan in zip(sorted(files_2080),sorted(files_a6000),sorted(files_titan)):
        base_2080, _ = os.path.splitext(file_2080)
        base_titan, _ = os.path.splitext(file_titan)
        base_a6000, _ = os.path.splitext(file_a6000)

        splitted_file = file_a6000.split("_")
        dict = return_dict(splitted_file)


        df_file_a6000 = pd.read_csv(directory_a6000+"/"+file_a6000)
        df_file_titan = pd.read_csv(directory_titan+"/"+file_titan)
        df_file_2080 = pd.read_csv(directory_2080+"/"+file_2080)

        tables = [df_file_2080,df_file_a6000,df_file_titan]
        names = ["2080","a6000","titan"]
        bases = [base_2080,base_a6000,base_titan]
        title = get_title_from_dict(dict)

        espace = 1 
        if dict["per-image"] == "True": espace = 5

        for i in range(len(tables)):
            le_max_time = tables[i].time.max()
            plt.figure(figsize=(10,6))

            if dict["only-cpu"] == "False":
                le_max_utilisation_gpu = tables[i].utilization_gpu.max()
                le_min_utilisation_gpu = tables[i].utilization_gpu.min()
                plt.plot(tables[i].time,tables[i].utilization_gpu)
                plt.xticks(np.arange(0, le_max_time+1))
                plt.yticks(np.arange(le_min_utilisation_gpu, le_max_utilisation_gpu+1))
                plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both'))
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both'))
                plt.xlabel("Temps (s)")
                plt.ylabel("Utilisation GPU % ")
                plt.suptitle(title,fontsize=10)
                plt.title(f"GPU {names[i]} utilisation du GPU % en fonction du temp (s)")
                plt.savefig(f"plot_images/one_batch_comparaison/{names[i]}/{modele_name}/gpu%_vs_preds_{bases[i]}.jpg", dpi=300)
                plt.close()

            plt.figure(figsize=(10,6))

            le_max_utilisation_gpu_watt = tables[i].poer_dra.max()
            le_min_utilisation_gpu_watt = tables[i].poer_dra.min()
            plt.plot(tables[i].time,tables[i].poer_dra)
            plt.xticks(np.arange(0, le_max_time+1))
            plt.yticks(np.arange(le_min_utilisation_gpu_watt, le_max_utilisation_gpu_watt+1))
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both'))
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both'))
            plt.xlabel("Temps (s)")
            plt.ylabel("Utilisation GPU Watt ")
            plt.suptitle(title,fontsize=10)
            plt.title(f"GPU {names[i]} utilisation du GPU Watt en fonction du temp (s)")
            plt.savefig(f"plot_images/one_batch_comparaison/{names[i]}/{modele_name}/gpuWatt_vs_preds_{bases[i]}.jpg", dpi=300)
            plt.close()

            plt.figure(figsize=(10,6))
            
            "Utilisation du CPU en % en fonction du temp de prédiction"
            le_max_utilisation_cpu = tables[i].cpu_usage.max()
            le_min_utilisation_cpu = tables[i].cpu_usage.min()
            plt.plot(tables[i].time,tables[i].cpu_usage)
            plt.xticks(np.arange(0, le_max_time+1))
            plt.yticks(np.arange(le_min_utilisation_cpu, le_max_utilisation_cpu+1))
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both'))
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both'))
            plt.xlabel("Temps (s)")
            plt.ylabel("Utilisation CPU %")
            plt.suptitle(title,fontsize=10)
            plt.title(f"utilisation du CPU % en fonction du temp (s)")
            plt.savefig(f"plot_images/one_batch_comparaison/{names[i]}/{modele_name}/cpu%_vs_preds_{bases[i]}.jpg", dpi=300)
            plt.close()
