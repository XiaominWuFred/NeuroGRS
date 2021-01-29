# NeuroGRS
NeuroGRS uses greedy structure searching based structural pruning method GRS and
state of art unstructured pruning method TQ to optimize given machine learning 
models and pick an best optimized model according to given metrics for calcium-
imaging-based predictive modeling. NeuroGRS system software is design based on 
dataflow modeling format, which is well known and used in signal processing 
field. Dataflow modeling format has many advantages in system design.  it 
provides well-defined in-terfaces between functional components, and 
exposesimportant forms of high-level application structure that are useful 
for reliable and efficient implementation inhardware or software. Details of 
NeuroGRS please refer to the following paper:
paper link.

## Dataflow based NeuroGRS software
The software is designed in dataflow modeling format with python3 programming language.   
### Inputs
NeuroGRS takes a set of Deep Neural Network (DNN) models and dataset contain features (X) and labels (Y) as inputs.  

#### Model Input
Set of DNN models M={m_1,m_2,...,m_n} are limited to MLP and CNN models.  
The set M can have only one type of model, all MLP models or all CNN models 
The set M can also contain both types of models.  
Models should be created by modifying the following .py file:  
NeuroGRS/src/util/modelPre.py  
Create your own original model by using keras format. 
Link to Keras:  

User should provide a unique model name for each model and save the model name with the model with the following instructions inside modelPre.py:  
```
        self.model_name.append('model_name')
        saveModel(model_object,'model_name')
```
An example for saving a model object cnnmulti with name 'cnnmulti':
```
        self.model_name.append('cnnmulti')
        saveModel(cnnmulti,'cnnmulti')
```
User has to provide name for each layer.  

#### Dataset Input
Input X={F_1,F_2,...,F_m} is a set of m input features.  

One input feature for MLP model mlpF_i={f_1,f_2,...,f_n} is a set of feature values in n dimension.  
The input X for MLP model should be saved as a 2D numpy array in Python3 with shape:(m,n)  

One input feature for CNN model cnnF_i is a 2D numpy array in Python3 with shape: (D_Row,D_Column)  
The input X for CNN model should be saved as a list of 2D numpy array or a 3D numpy array in Python3 with shape: (m,D_Row,D_Column)  

Input Y={l_1,l_2,...,l_m} is a set of m labels for m input features in X.  
It should be saved as a 1D numpy array in Python3 with shape: (m,1)  

X and Y should be fully shuffled with consistent indexing, and then separated into three parts: trainX, valX, testX; trainY, valY, testY with a ratio (usually 8:1:1)  

Since the dataset format and naming can be different for each user, we give flexibility to user to decide how to load their dataset into required format.  
Users are expected to prepare inputs by modifying the following .py file:  
NeuroGRS/src/util/datasetPre.py  

The datasetPre.py file was filled in with default codes for preparing X and Y from our dataset of our application case.  
User should first erase all default custom variables list in __init__(self,args) and assign required variables according to the following instructions:    

For MLP models only:
modify function run()  
assign the following MLP related variables (default as None) with prepared dataset mentioned above  
```
	self.mlptrainX = None
        self.mlptrainY = None
        self.mlpvalX = None
        self.mlpvalY = None
        self.mlptestX = None
        self.mlptestY = None
```
Specify file name for output file naming to the following variable (default as None):  
```
		self.file_name = None
```
Leave any CNN related varibale as default value (None)

For CNN models only:
assign the following CNN related variables (default as None) with prepared dataset mentioned above
```
	self.cnntrainX = None
        self.cnntrainY = None
        self.cnnvalX = None
        self.cnnvalY = None
        self.cnntestX = None
        self.cnntestY = None
```
Specify file name for output file naming to the following variable (default as None):
```
		self.file_name = None
```
provide input dimension to CNN model to the following variables:
```
        self.D_Row = None
        self.D_Column = None
```
Leave any MLP related varibale as default value (None)

FOr combined CNN and MLP models:
Assign proper value according to instruction above to both MLP related variables and CNN related variables.  

Custom variables can be added to __init__(self,args) as long as required variables mentioned above are provided in required format.
Additional Custom inputs can be added into args, which is a python list default to None.  

Change the inputs to the call of constructor in NeuroGRS/src/graph/neuroGRS_graph.py  
```
	dataPre = datasetPre([customInput_1, customInput_2,...,customInput_n])  
```

### Run NeuroGRS
After prepared all dataset and model inputs, modify required argument in the following driver file:  
```
NeuroGRS/src/driver/driver.sh
```
Add arguments to the following instruction:  
```
python3 neuroGRS_graph.py
```
The first argument is EUP, 1 for enable unstructured pruning (TQ), 0 for unstructured pruning GRS only.  
Begin with the second argument, required custom arguments should be put. 
An example with unstructured pruning TQ enabled and custom arguments for default datasetPre.py for our application case: 
second argument is version mark, third argument is dataset name to choose among different datasets.   
```
python3 neuroGRS_graph.py 1 v 04e1
```
NeuroGRS is ready to run with following commands:  
```
cd NeuroGRS/src/driver/
./runme 
```

### Outputs
NeuroGRS software provides the following output files:  
#### Models and weights
Saved original model information and pruned model information and model weights after TQ if enabled.  
Models are saved as .json file, weights are saved as .h5 file.  
Models and weights outputs are saved in:  
```
NeuroGRS/outputs/modelinfo/
```
With naming contain file_name provided in datasetPre.py and model_name provided in modelPre.py  
#### GRS pruning trend plots
Every pruning step of GRS is saved as a plot. 
The plot will be saved in:
```
NeuroGRS/outputs/plots/
```
With naming contain file_name provided in datasetPre.py and model_name provided in modelPre.py  
#### GRS statistics
Overall statistics of pruning is saved as .csv file with following information:  
,GRS_TestAcc(lost%),GRS_ValAcc(lost%),GRS_FLOPs(reduced%),GRS_Paras(reduced%),Original structure,GRS_found structure  
The .csv file will be saved under:  
```
NeuroGRS/outputs/prunedM/
```
With naming contain file_name provided in datasetPre.py and model_name provided in modelPre.py 
#### Runtime Logs
Logs contain each steps of TQ pruning process is recored in .txt file and will be saved under:  
```
NeuroGRS/outputs/runInfo/
```
With naming contain file_name provided in datasetPre.py and model_name provided in modelPre.py 


## Dataflow based inference system for structured pruned model from GRS
