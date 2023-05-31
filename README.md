# Benchmarking on Tiny Imagenet

#### We tried to evaluate the performance of 3 models on the Tiny Imagenet Dataset depending on multiples factor 
#### Our goal was to find the max FLOPS that a model should have to be faster on CPU thanks Quantization compared to a GPU

#### Factors:
<pre>
Image Size <br>
Quantization: <br>
  float16 <br>
  Dynamic Range <br>
  Full Integer  <br>
Model: <br>
  MobileNet <br>
  ResNet50 <br>
  EfficientNetB0 <br>
GPU ON/OFF ( the GPU used was a TITAN RTX ) <br>

</pre>
The result show that if a model has less than 1.5 FLOPS with quantization is faster with a CPU than the same model with a GPU.

## Run the Experiments
NOTE: scripts for the experiments are in the running_scripts folder.  
#### 1.0 Training. 
First you'll need to re-train one of the model using the train_save.py script, check out the code to see the name of the args.  
#### 2.0 Optimization. 
Then run the load_quantize.py script to quantize the model that you trained. as the first script you'll need to check the name of the args.  
#### 3.0 Evaluation. 
To perform evaluation you'll need to execute the main.py script, the script require multiple params that you should give ( see args.py). It will automatically make evaluate the model in "real time" with batch size 1 and at the same time register useful information about the consumations of the GPU/CPU and time...  

## Visualization
NOTE: script for the visualization is in plot_scripts folder.  
### 1.0 Visualize. 
To visualise the csv file that has been saved from the evaluation you'll need to run the main.py scripts.  
<br> I Added manually the acc and flops columns for each csv file </br>. 