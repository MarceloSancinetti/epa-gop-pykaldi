from pytorch_models import *
import numpy as np

def read_linear_params(file):
	global line
	linear_params = []
	while ']' not in line:
		line = file.readline()
		line_array = np.fromstring(line.replace(']',''), dtype=float, sep=' ')
		linear_params.append(line_array)
	linear_params = np.array(linear_params)
	return linear_params

def read_bias(file):
	global line
	line = file.readline()
	line = line.replace('<BiasParams>  [ ', '')
	line = line.replace(']', '')
	bias = np.fromstring(line, dtype=float, sep=' ')
	return bias

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
		#Read linear parameters
		components['lda']['linear_params'] = read_linear_params(chain_file)
		
		#Read bias
		components['lda']['bias'] = read_bias(chain_file)

	if '<ComponentName> tdnn1.affine' in line:
		components['tdnn1.affine'] = {}
		#If necessary, parse parameters such as learning rate here
		components['tdnn1.affine']['linear_params'] = read_linear_params(chain_file)
		components['tdnn1.affine']['bias'] = read_bias(chain_file)


  
print(components['lda']['bias'])
chain_file.close() 

hola = FTDNN()
print(hola)