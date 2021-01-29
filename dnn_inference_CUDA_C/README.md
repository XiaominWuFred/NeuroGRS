--author xiaomin wu    
--date 1/16/2020

cnnnsight: contain CUDA codes
dataset: the data set used in simple CNN example implementation
extractParas: extracted from python model, every inputs outputs from
		neuron network. 
lide_c_dnnlayers: wrapped lide_c version of simple CNN example. actors 
		are reusable
simpleCnn: python version of simple CNN example, and extracting scripts

How to run:
    CUDA version (CUDA10.0 CUDA standard library required):
	cd lide_c_dnnlayers/test_cuda/
	bash make.bash
	./driver
    C version:
	cd lide_c_dnnlayers/test_c/
	bash make.bash
	./driver

