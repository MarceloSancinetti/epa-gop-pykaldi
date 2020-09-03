from pytorch_models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_component_type(is_tdnnf):
	component_type = 'tdnn'
	if is_tdnnf:
		component_type = 'tdnnf'
	return component_type

def extract_batchnorm_count(string, layer_number, is_tdnnf = True):
	component_type = get_component_type(is_tdnnf)
	string = string.replace('<ComponentName> '+component_type+str(layer_number)+'.batchnorm <BatchNormComponent> <Dim> 1536 <BlockDim> 1536 <Epsilon> 0.001 <TargetRms> 1 <TestMode> F <Count> ', '')
	count = ''
	i = 0
	while string[i] != ' ':
		count = count + string[i]
		i = i+1
	return count

def read_linear_params(file):
	global line
	linear_params = []
	while ']' not in line:
		line = file.readline()
		line_array = np.fromstring(line.replace(']',''), dtype=float, sep=' ')
		linear_params.append(line_array)
	linear_params = np.array(linear_params)
	#print(linear_params)
	return linear_params

def read_vector(file, prefix, advance):
	global line
	if advance: 
		line = file.readline()
	line = line.replace(prefix, '')
	line = line.replace(']', '')
	if len(line) <= 1:
		#print(np.array([]))
		return np.array([])
	#print(line)
	vector = np.fromstring(line, dtype=float, sep=' ')
	#print(vector)
	return vector

def read_bias(file):
	return read_vector(file, '<BiasParams>  [ ', True)

def read_value_avg(file, layer_number, is_tdnnf = True):
	component_type = get_component_type(is_tdnnf)
	return read_vector(file, '<ComponentName> '+ component_type + str(layer_number) +'.relu <RectifiedLinearComponent> <Dim> 1536 <ValueAvg>  [ ', False)

def read_deriv_avg(file):
	return read_vector(file, '<DerivAvg>  [ ', False)

def read_stats_mean(file, layer_number, is_tdnnf = True):
	component_type = get_component_type(is_tdnnf)
	count = extract_batchnorm_count(line,layer_number, is_tdnnf)
	return read_vector(file, '<ComponentName> '+ component_type + str(layer_number) +'.batchnorm <BatchNormComponent> <Dim> 1536 <BlockDim> 1536 <Epsilon> 0.001 <TargetRms> 1 <TestMode> F <Count> '+ count +' <StatsMean>  [ ', False)

def read_stats_var(file):
	return read_vector(file, '<StatsVar>  [ ', False)

def read_affine_component(file):
	params_dict = {}
	params_dict['linear_params'] = read_linear_params(file)
	params_dict['bias'] = read_bias(file)
	return params_dict

def read_batchnorm_component(file, layer_number, is_tdnnf=True):
	params_dict = {}
	params_dict['stats_mean'] = read_stats_mean(file, layer_number, is_tdnnf=is_tdnnf)
	line = chain_file.readline()
	params_dict['stats_var'] = read_stats_var(file)
	return params_dict

def read_relu_component(file, layer_number, is_tdnnf=True):
	params_dict = {}
	params_dict['value_avg'] = read_value_avg(file, layer_number, is_tdnnf=is_tdnnf)
	line = chain_file.readline()
	params_dict['deriv_avg'] = read_deriv_avg(file)
	return params_dict

chain_file = open('final', 'r') 
 
components = {}
finished = False  
while not finished: 
  
	line = chain_file.readline() 

	if line == '</Nnet3>':
		finished = True

	if '<ComponentName> lda' in line:
		components['lda'] = read_affine_component(chain_file)

	if '<ComponentName> tdnn1.affine' in line:
		#If necessary, parse parameters such as learning rate here
		components['tdnn1.affine'] = read_affine_component(chain_file)

	if '<ComponentName> tdnn1.relu' in line:
		components['tdnn1.relu'] = read_relu_component(chain_file, 1, is_tdnnf=False)

	if '<ComponentName> tdnn1.batchnorm' in line:
		components['tdnn1.batchnorm'] = read_batchnorm_component(chain_file, 1, is_tdnnf=False)

	#No tdnn1.dropout yet


	#Reads parameters for layers 2-17
	for layer_number in range(2, 18):
		layer_name = 'tdnnf'+str(layer_number)

		if '<ComponentName> ' + layer_name + '.affine' in line:
			line = chain_file.readline() #Skip unnecessary line
			components[layer_name + '.affine'] = read_affine_component(chain_file)

		if '<ComponentName> ' + layer_name + '.relu' in line:
			components[layer_name + '.relu'] = read_relu_component(chain_file, layer_number)

		if '<ComponentName> ' + layer_name + '.batchnorm' in line:
			components[layer_name + '.batchnorm'] = read_batchnorm_component(chain_file, layer_number)

		if '<ComponentName> ' + layer_name + '.linear' in line:
			line = chain_file.readline() #Skip unnecessary line
			components[layer_name + '.linear'] = read_affine_component(chain_file)

		#No tdnnfx.dropout yet
		#No tdnnfx.noop yet
	
	'''
	if '<ComponentName> prefinal-chain.affine' in line:
		components['prefinal-chain.affine'] = read_affine_component(chain_file)

	if '<ComponentName> prefinal-chain.linear' in line:
		components['prefinal-chain.linear'] = {}
		components['prefinal-chain.linear']['linear_params'] = read_linear_params(chain_file)
	'''

	if '<ComponentName> prefinal-xent.affine' in line:
		components['prefinal-xent.affine'] = read_affine_component(chain_file)

	if '<ComponentName> prefinal-xent.linear' in line:
		components['prefinal-xent.linear'] = {}
		components['prefinal-xent.linear']['linear_params'] = read_linear_params(chain_file)

	if '<ComponentName> output-xent.affine' in line:
		components['output-xent.affine'] = read_affine_component(chain_file)

print("Components")
print(components['lda']['linear_params'].shape)
print(components['lda']['bias'].shape)
print(components['tdnn1.affine']['linear_params'].shape)
print(components['tdnn1.affine']['bias'].shape)
for n in range(2, 17):
	print('Layer '+str(n))
	print(components['tdnnf'+str(n)+'.linear']['linear_params'].shape)
	print(components['tdnnf'+str(n)+'.linear']['bias'].shape)
	print(components['tdnnf'+str(n)+'.affine']['linear_params'].shape)
	print(components['tdnnf'+str(n)+'.affine']['bias'].shape)
	
print(components['prefinal-xent.affine']['linear_params'].shape)
print(components['prefinal-xent.linear']['linear_params'].shape)
print(components['output-xent.affine']['linear_params'].shape)

ftdnn = FTDNN()
print(ftdnn.state_dict)

#state_dict = {}

#tate_dict['tdnn1'] = {}
#state_dict['tdnn1']['kernel.weight'] = torch.from_numpy(components['tdnn1.affine']['linear_params'])[:,:,None]
#state_dict['tdnn1']['kernel.bias'] = torch.from_numpy(components['tdnn1.affine']['bias'])
#state_dict['tdnn1']['bn.running_mean'] = torch.from_numpy(components['tdnn1.batchnorm']['stats_mean'])
#state_dict['tdnn1']['bn.running_var'] = torch.from_numpy(components['tdnn1.batchnorm']['stats_var'])

chain_file.close() 

