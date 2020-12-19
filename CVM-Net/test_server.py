import validate_nyc
from validate_nyc import TestImage
k = 5
image = '/scratch1/surabhi/crossview_localisation/src/Data/nyc/streetview/7.jpg'
ti = TestImage(k,image)
sat_paths, gnd_paths, gps = ti.find_knn()
print(gps)
# image = '/scratch1/surabhi/crossview_localisation/src/Data/sydney_orignal/streetview/8.jpg'
# ti_ = TestImage(k,image)
# sat_paths, gnd_paths, gps = ti_.find_knn()
