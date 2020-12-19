A path kml, describing the path is required. Generate a kml file with Google My Maps.  

For streetview concatnation (to create 360 degree panorama), use img_concat.py 
$ python image_concat.py

Download aerial view images only if the corresponding streetview is present. Number them according to order. For this, run img_order.py
$ python img_order.py

Do proper renaming and store the indices of points:
$ python rename.py >> index.txt

form train and test csv: 
$ python form_train.py

To transfer it to remote server: These are just examples. you will have to transfer these files/folders in the corresponding <your_name> directory and inside the src folder.
$ scp -r NY_pano_order/*.jpg tokekarwks00.umiacs.umd.edu:/scratch1/surabhi/crossview_localisation/src/Data/nyc/streetview
$ scp -r NY_sat_order/*.jpg tokekarwks00.umiacs.umd.edu:/scratch1/surabhi/crossview_localisation/src/Data/nyc/satellite
$ scp -r *.csv tokekarwks00.umiacs.umd.edu:/scratch1/surabhi/crossview_localisation/src/Data/nyc/splits
