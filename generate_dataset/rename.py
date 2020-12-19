import os
from PIL import Image

os.chdir('../../New_York_sat')
sat_path = os.getcwd()
os.chdir('../New_York_pano')
pano_path = os.getcwd()
os.chdir('../NY_pano_order')
pano_save = os.getcwd()
os.chdir('../NY_sat_order')
sat_save = os.getcwd()

file_num = 0

for subdir, dirs, files in os.walk(pano_path):
	files.sort(key=lambda f: int(filter(str.isdigit, f)))
	# files.sort()
	print(len(files))
	for file in files:
		print(file)
		file_len = len(file)-4
		current_ind = file[:file_len]
		os.rename(os.path.join(dirpath, file), os.path.join(pano_save, str(file_num) + '.jpg'))
		Image.open(sat_path+os.sep+str(current_ind)+'.png').convert('RGB').save(sat_save+os.sep+str(file_num)+'.jpg')
		# os.rename(os.path.join(dirpath, file), os.path.join(pano_save, str(file_num) + '.jpg'))
		file_num += 1

# to store indices
# for subdir, dirs, files in os.walk(sat_path):
# 	files.sort(key=lambda f: int(filter(str.isdigit, f)))
# 	# files.sort()
# 	# print(len(files))
# 	for file in files:
# 		# print(file)
# 		file_len = len(file)-4
# 		current_ind = file[:file_len]
# 		print(current_ind)

