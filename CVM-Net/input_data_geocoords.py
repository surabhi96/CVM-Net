import cv2
import random
import numpy as np
import os
import time
import logging

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


os.chdir(os.getcwd())

class TestData:
    def __init__(self, Data_folder, image_path):

        print('TestData')

        cwd = os.getcwd()
        os.chdir('../Data/'+Data_folder+'/')
        self.img_root = os.getcwd()
        os.chdir('../../CVM-Net')

        # self.test_list = self.img_root + '/splits/' + Test_file
        #
        # print('TestData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for testing
        self.id_test_list = []
        self.image_path = image_path
        # read all the test data
        print('satellite/0.jpg,'+self.image_path+', ')
        line = 'satellite/0.jpg,'+self.image_path+', '
        data = line.split(',')
        pano_id = (data[0].split('/')[-1]).split('.')[0]
        # satellite filename, streetview filename
        self.id_test_list.append([data[0], data[1], pano_id])
        # idx += 1

        self.test_data_size = len(self.id_test_list)

    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None  # done iterating with all the test sample batches
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            # readjust batch size for the final iteration
            batch_size = self.test_data_size - self.__cur_test_id
            # this is the size of the input ground image
        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype=np.float32)
        # this is the size of input to the identical network dealing with satellite image
        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)

        # iterate over the length of a batch
        for i in range(batch_size):
            # this is the image index
            img_idx = self.__cur_test_id + i

            # SATELLITE ; start from the first image
            sat_path = self.img_root + '/' + self.id_test_list[img_idx][0]
            img = cv2.imread(self.img_root + '/' + self.id_test_list[img_idx][0])
            # print(img)
            # only satellite image is reshaped
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            ###### manipulate rgb channels???
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            # keep storing images in the satellite image batch
            batch_sat[i, :, :, :] = img

            # GROUND
            gnd_path = self.id_test_list[img_idx][1]
            # print("gnd_path")
            # print(gnd_path)
            img = cv2.imread(gnd_path)

            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            # do the same storage for the ground view images
            batch_grd[i, :, :, :] = img
            # print("sat_path")
            # print(sat_path)
        # update the cur_test_id to the next batch
        # the value of cur_test_id is always k*batch_size where k is 1,2,3...
        self.__cur_test_id += batch_size
        # return the satellite and ground batch with images contained!
        return batch_sat, batch_grd

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0

# for the descriptors; only the test file; only one file
class ValidateData:

    def __init__(self, Data_folder, Test_file):

        print('ValidateData')

        cwd = os.getcwd()
        os.chdir('../Data/'+Data_folder+'/')
        self.img_root = os.getcwd()
        os.chdir('../../CVM-Net')

        self.test_list = self.img_root + '/splits/' + Test_file

        print('TestData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for testing
        self.id_test_list = []
        # read all the test data
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename
                self.id_test_list.append([data[0], data[1], pano_id])
                idx += 1
        self.test_data_size = len(self.id_test_list)
        print('TestData::__init__: load', self.test_list, ' data_size =', self.test_data_size)

    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None  # done iterating with all the test sample batches
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            # readjust batch size for the final iteration
            batch_size = self.test_data_size - self.__cur_test_id
            # this is the size of the input ground image
        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype=np.float32)
        # this is the size of input to the identical network dealing with satellite image
        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)

        # iterate over the length of a batch
        for i in range(batch_size):
            # this is the image index
            img_idx = self.__cur_test_id + i

            # SATELLITE ; start from the first image
            sat_path = self.img_root + '/' + self.id_test_list[img_idx][0]
            img = cv2.imread(self.img_root + '/' + self.id_test_list[img_idx][0])
            # only satellite image is reshaped
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            ###### manipulate rgb channels???
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            # keep storing images in the satellite image batch
            batch_sat[i, :, :, :] = img

            # GROUND
            gnd_path = self.img_root + '/' + self.id_test_list[img_idx][1]
            # print("gnd_path")
            # print(gnd_path)
            img = cv2.imread(gnd_path)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            # do the same storage for the ground view images
            batch_grd[i, :, :, :] = img
            # print("sat_path")
            # print(sat_path)
        # update the cur_test_id to the next batch
        # the value of cur_test_id is always k*batch_size where k is 1,2,3...
        self.__cur_test_id += batch_size  # return the satellite and ground batch with images contained!
        return batch_sat, batch_grd

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0

class InputData:

    def __init__(self):

        print("InputData")
        cwd = os.getcwd()
        os.chdir('../Data/sydney_dense/')
        self.img_root = os.getcwd()
        self.valid_root = self.img_root
        os.chdir('../../CVM-Net')

        self.train_list = self.img_root + '/splits/train-19zl.csv'
        self.test_list = self.img_root + '/splits/test-19zl.csv'
        self.train_test = self.img_root + '/splits/train-test.csv'
        self.train_geo = self.img_root + '/splits/train-geo.csv'
        self.test_geo = self.img_root + '/splits/test-geo.csv'
        
        # this is to load the train dataset
        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                # just the jpg numbers
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([data[0], data[1], pano_id])  # pano_id is just a number
                self.id_idx_list.append(idx)  # just integers 0,1,2 .. with every line in split file
                idx += 1
        self.data_size = len(self.id_list)  # 35532 for cvusCVM-Net-I_syd_originala # length of train data
        print('InputData::__init__: load', self.train_list, ' data_size =', self.data_size)

        # This is to load the test dataset
        print('InputData::__init__: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            li = 0
            for line in file:
                # print(line)
                li += 1
                # print('li ,',li)
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([data[0], data[1], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)  # 8884 for cvusa
        print('InputData::__init__: load', self.test_list, ' data_size =', self.test_data_size)

        # This is to load the entire dataset
        print('InputData::__init__: load %s' % self.train_test)
        self.__cur_tt_id = 0  # for training
        self.id_tt_list = []
        self.id_tt_idx_list = []
        with open(self.train_test, 'r') as file:
            idx = 0
            li = 0
            for line in file:
                # print(line)
                li += 1
                # print('li ,',li)
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_tt_list.append([data[0], data[1], pano_id])
                self.id_tt_idx_list.append(idx)
                idx += 1
        self.tt_data_size = len(self.id_tt_list)  # 8884 for cvusa
        print('InputData::__init__: load', self.train_test, ' data_size =', self.tt_data_size)

        self.train_geocoord = []
        self.test_geocoord = []
        with open(self.train_geo, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                
                self.train_geocoord.append([data[0], data[1]])  # pano_id is just a number
                
                idx += 1

        with open(self.test_geo, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                
                # satellite filename, streetview filename, pano_id
                self.test_geocoord.append([data[0], data[1]])
            
                idx += 1
    # this is used while testing
    def next_batch_scan(self, batch_size):

        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id
        # this is the size of the input ground image
        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype=np.float32)
        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)
        batch_coords = np.zeros([batch_size, 2], dtype=np.float32)
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i
            satellite_view_img = self.valid_root + '/' + self.id_test_list[img_idx][0]
            print(satellite_view_img)
            # satellite
            # img = cv2.imread(self.img_root + '/' + self.id_test_list[img_idx][0])
            # for combined (and ordered) test list
            img = cv2.imread(satellite_view_img)
            # only satellite image is reshaped
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # ground
            # img = cv2.imread(self.img_root + '/' + self.id_test_list[img_idx][1])
            ground_view_img = self.valid_root + '/' + self.id_test_list[img_idx][1]
            img = cv2.imread(ground_view_img)
            # print('hei')

            # print(ground_view_img)
            try:
                img = img.astype(np.float32)
            except:
                print(img)


            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img

            batch_coords[i,:] = self.test_geocoord[i,:]

        self.__cur_test_id += batch_size
        print('returning input')
        return batch_sat, batch_grd, batch_coords

        # this is used while training
    def next_pair_batch(self, batch_size):
            if self.__cur_id == 0:
                for i in range(20):
                    # shuffle the id list 20 times
                    random.shuffle(self.id_idx_list)

            if self.__cur_id + batch_size + 2 >= self.data_size:
                self.__cur_id = 0
                return None, None

            batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)
            batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype=np.float32)
            batch_coords = np.zeros([batch_size, 2], dtype=np.float32)

            i = 0
            batch_idx = 0
            while True:
                if batch_idx >= batch_size or self.__cur_id + i >= self.data_size:
                    break

                img_idx = self.id_idx_list[self.__cur_id + i]
                i += 1

                # satellite
                img_path = self.img_root + '/' + self.id_list[img_idx][0]
                img = cv2.imread(img_path)
                # print("img")
                print(img_path)
                if img is None or img.shape[0] != img.shape[1]:
                    print(img_path)
                    print(
                        'InputData::next_pair_batch: read fail: %s, %d, ' % (
                            self.img_root + '/' + self.id_list[img_idx][0], i),
                        img.shape)
                    continue
                rand_crop = random.randint(1, 748)
                if rand_crop > 512:
                    start = int((750 - rand_crop) / 2)
                    img = img[start: start + rand_crop, start: start + rand_crop, :]
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                rand_rotate = random.randint(0, 4) * 90
                rot_matrix = cv2.getRotationMatrix2D((256, 256), rand_rotate, 1)
                img = cv2.warpAffine(img, rot_matrix, (512, 512))
                img = img.astype(np.float32)
                # img -= 100.0
                img[:, :, 0] -= 103.939  # Blue
                img[:, :, 1] -= 116.779  # Green
                img[:, :, 2] -= 123.6  # Red
                batch_sat[batch_idx, :, :, :] = img
                # ground
                img_path = self.img_root + '/' + self.id_list[img_idx][1]
                img = cv2.imread(img_path)
                if img is None or img.shape[0] != 224 or img.shape[1] != 1232:
                    print('buggy %s ' % (self.img_root + '/' + self.id_list[img_idx][1]))
                    print(
                        'InputData::next_pair_batch: read fail: %s, %d, ' % (
                            self.img_root + '/' + self.id_list[img_idx][1], i),
                        img.shape)
                    continue
                img = img.astype(np.float32)
                # img -= 100.0
                img[:, :, 0] -= 103.939  # Blue
                img[:, :, 1] -= 116.779  # Green
                img[:, :, 2] -= 123.6  # Red
                batch_grd[batch_idx, :, :, :] = img

                batch_coords[batch_idx,0] = self.test_geocoord[batch_idx][0]
                batch_coords[batch_idx, 1] = self.test_geocoord[batch_idx][1]

                batch_idx += 1

            self.__cur_id += i

            return batch_sat, batch_grd, batch_coords

    # this is to read the combined train and test dataset
    def next_tt_scan(self, batch_size):

        if self.__cur_tt_id >= self.tt_data_size:
            self.__cur_tt_id = 0
            return None, None
        elif self.__cur_tt_id + batch_size >= self.tt_data_size:
            batch_size = self.tt_data_size - self.__cur_tt_id
        # this is the size of the input ground image
        batch_grd = np.zeros([batch_size, 224, 1232, 3], dtype=np.float32)
        batch_sat = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)
        for i in range(batch_size):
            img_idx = self.__cur_tt_id + i
            satellite_view_img = self.valid_root + '/' + self.id_tt_list[img_idx][0]
            print(satellite_view_img)
            # satellite
            # img = cv2.imread(self.img_root + '/' + self.id_test_list[img_idx][0])
            # for combined (and ordered) test list
            img = cv2.imread(satellite_view_img)
            # only satellite image is reshaped
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # ground
            # img = cv2.imread(self.img_root + '/' + self.id_test_list[img_idx][1])
            ground_view_img = self.valid_root + '/' + self.id_tt_list[img_idx][1]
            print(ground_view_img)

            img = cv2.imread(ground_view_img)
            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img

        self.__cur_tt_id += batch_size
        print('returning input')
        return batch_sat, batch_grd

    # get train dataset size
    def get_dataset_size(self):
        return self.data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def get_tt_dataset_size(self):
        return self.tt_data_size

    def reset_scan(self):
        self.__cur_test_idd = 0

