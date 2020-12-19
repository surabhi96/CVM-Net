import sys
from PIL import Image
import os
import time
dirpath = os.getcwd()
import numpy as np
import requests
from fastkml import kml
import time
import io

images_ = []
img_num = 0
img = 0

def distance(lat1, lon1, lat2, lon2):
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

points=[]
# for p in data.coords:
#     points.append([p[1],p[0]])
os.chdir('../')
with open(os.getcwd()+'/New_York.csv', 'r') as file:
    for line in file:
        data = line.split(',')
        points.append([data[1],data[0]])
# print(points)

p_len = len(points)
# print(p_len)

os.chdir('../../independent_study/New_York_final/')
pano_path = os.getcwd()
os.chdir('../New_York_pano/')
pano_save = os.getcwd()
os.chdir('../New_York_sat/')
sat_save = os.getcwd()

# go through files in pano (query)
for subdir, dirs, files in os.walk(pano_path):
    files.sort()
    for file in files:
        # print(file)
        filepath = subdir + os.sep + file
        if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):
            # file_len = len(file)-6
            file_len = len(file)-4
            # query image
            file_name = file[:file_len]
            items = file_name.split('_')
            # print(file_name)
            # print(items[1], items[2])
            link = 'https://maps.googleapis.com/maps/api/staticmap?center='+items[1]+','+items[2]+'&zoom=20&size=512x512&tilt=90&maptype=satellite&key=AIzaSyDJjxJOwPhpG_RV3gNXP3ULOQpSRLkleKI'
            # print(link)
            r = requests.get(link)
            
            ind = 0 
            if(r.content):
                print("file_there!")
                for k in range (p_len):
                    # print(items[1], items[2])
                    if((items[1] == str(points[k][0])) and (items[2] == str(points[k][1]))):
                        print('{} found'.format(k))
                        ind = k

                        # os.chdir('../../New_York_sat')
                        f = open(str(ind)+'.png', 'wb')
                        
                        f.write(r.content)
                    
                        f.close()

                        f = Image.open(filepath) 
                        
                        f = f.save(pano_save+os.sep+str(ind)+'.jpg') 
                        break
            else:
                print('file not found')
                # print(file_name)
                continue
            
            # time.sleep(1)
