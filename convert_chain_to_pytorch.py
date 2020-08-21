from pytorch_models import *
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
	print(linear_params)
	return linear_params

def read_vector(file, prefix, advance):
	global line
	if advance: 
		line = file.readline()
	line = line.replace(prefix, '')
	line = line.replace(']', '')
	if len(line) <= 1:
		print(np.array([]))
		return np.array([])
	#print(line)
	vector = np.fromstring(line, dtype=float, sep=' ')
	print(vector)
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

# Using readline() 
chain_file = open('final', 'r') 
 
components = {}
finished = False  
while not finished: 
  
	line = chain_file.readline() 

	if line == '</Nnet3>':
		finished = True

	if '<ComponentName> lda' in line:
		components['lda'] = {}
		components['lda']['linear_params'] = read_linear_params(chain_file)
		components['lda']['bias'] = read_bias(chain_file)

	if '<ComponentName> tdnn1.affine' in line:
		components['tdnn1.affine'] = {}
		#If necessary, parse parameters such as learning rate here
		components['tdnn1.affine']['linear_params'] = read_linear_params(chain_file)
		components['tdnn1.affine']['bias'] = read_bias(chain_file)

	if '<ComponentName> tdnn1.relu' in line:
		components['tdnn1.relu'] = {}
		components['tdnn1.relu']['value_avg'] = read_value_avg(chain_file,1, is_tdnnf=False)
		line = chain_file.readline()
		components['tdnn1.relu']['deriv_avg'] = read_deriv_avg(chain_file)

	if '<ComponentName> tdnn1.batchnorm' in line:
		components['tdnn1.batchnorm'] = {}
		components['tdnn1.batchnorm']['stats_mean'] = read_stats_mean(chain_file, 1, is_tdnnf=False)
		line = chain_file.readline()
		components['tdnn1.batchnorm']['stats_var'] = read_stats_var(chain_file)

	#No tdnn1.dropout yet


	#Reads parameters for layers 2-17
	for layer_number in range(2, 17):
		layer_name = 'tdnnf'+str(layer_number)

		component_type = '.affine'
		if '<ComponentName> ' + layer_name + component_type in line:
			components[layer_name + component_type] = {}
			line = chain_file.readline()
			components[layer_name + component_type]['linear_params'] = read_linear_params(chain_file)
			components[layer_name + component_type]['bias'] = read_bias(chain_file)

		component_type = '.relu'
		if '<ComponentName> ' + layer_name + component_type in line:
			components[layer_name + component_type] = {}
			components[layer_name + component_type]['value_avg'] = read_value_avg(chain_file,layer_number)
			line = chain_file.readline()
			components[layer_name + component_type]['deriv_avg'] = read_deriv_avg(chain_file)

		component_type = '.batchnorm'
		if '<ComponentName> ' + layer_name + component_type in line:
			components[layer_name + component_type] = {}
			components[layer_name + component_type]['stats_mean'] = read_stats_mean(chain_file, layer_number)
			line = chain_file.readline()
			components[layer_name + component_type]['stats_var'] = read_stats_var(chain_file)

		component_type = '.linear'
		if '<ComponentName> ' + layer_name + component_type in line:
			components[layer_name + component_type] = {}
			line = chain_file.readline()
			components[layer_name + component_type]['linear_params'] = read_linear_params(chain_file)
			components[layer_name + component_type]['bias'] = read_bias(chain_file)


		#No tdnnfx.dropout yet
		#No tdnnfx.noop yet



  
chain_file.close() 

