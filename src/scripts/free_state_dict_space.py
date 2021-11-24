import sys
import os
import re

if __name__ == '__main__':
	state_dict_dir_path = sys.argv[1]
	for state_dict_name in os.listdir(state_dict_dir_path):
		state_dict_name_split = re.split('[. -]',state_dict_name)
		
		if "swa" in state_dict_name_split[-2]:
			epoch = int(state_dict_name_split[-2].split("_")[0])
		else:
			epoch = int(state_dict_name_split[-2]) 

		if epoch % 25 != 0:
			full_path = os.path.join(state_dict_dir_path, state_dict_name)
			os.system("rm " + full_path)