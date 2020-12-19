import sys
from PIL import Image
import os
import time
import re
from glob import glob

dirpath = os.getcwd()

os.chdir('../../../new_york_pano/')
dirpath = os.getcwd()
os.chdir('../independent_study/New_York_final/')
save_path_ = os.getcwd()

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
	 
images_ = []
images=[]
file_val = []
file_names = []
img_num = 0
img = 0

for subdir, dirs, files in os.walk(dirpath):
	files.sort()

	for file in files:
		# print(file)
		filepath = subdir + os.sep + file
		if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):
				
			img_num += 1
			# print(images)
			file_len = len(file)-6
			file_name = file[:file_len]
			# time.sleep(2)
			file_names.append(file_name)
			images_.append(filepath)
			images.append(file)

			if (img_num == 4):
				if (file_names[0] == file_names[1] == file_names[2] == file_names[3]):
					# print(images)
					img_num = 0

					# print(images_)
					images = [Image.open(x) for x in images_]
					widths, heights = zip(*(i.size for i in images))

					total_width = sum(widths)

					max_height = max(heights)
					new_im = Image.new('RGB', (total_width, max_height))

					x_offset = 0
							
					for im in images:
						new_im.paste(im, (x_offset,0))
						x_offset += im.size[0]
					new_im = new_im.resize((1232, 224))
					# save_path = dirpath+'/dataset_final/street/'+str(img)+'.jpg'
					
					save_path = save_path_+os.sep+str(file_name)+'.jpg'
					# print(save_path)
					# time.sleep(2)
					new_im.save(save_path)
					img += 1
					images_=[]
					file_names = []
				else:
					print(file_names)
