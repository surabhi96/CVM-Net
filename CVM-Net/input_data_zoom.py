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

            # GROUNDself.geo_coordinates
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
#
class InputData:

    def __init__(self, radius=2):

        print("InputData")
        cwd = os.getcwd()
        os.chdir('../Data/sydney_dense/')
        self.img_root = os.getcwd()
        self.valid_root = self.img_root
        os.chdir('../../CVM-Net')

        self.train_list = self.img_root + '/splits/train-zoom.csv'
        self.test_list = self.img_root + '/splits/test-zoom.csv'
        # self.train_test = self.img_root + '/splits/train-test.csv'
        self.train_sat_list = self.img_root + '/splits/sat-zoom.csv'

        # this is to load the train dataset
        print('InputData::__init__: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        # satellite filename, streetview filename, pano_id
        self.id_list = []
        # Just numbers until length of train set
        self.id_idx_list = []
        # Pano id's
        self.pano_id_list = []
        pano_id_val = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                # just the jpg numbers
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([data[0], data[1], pano_id])  # pano_id is just a number
                self.id_idx_list.append(idx)  # just integers 0,1,2 .. with every line in split file
                self.pano_id_list.append(pano_id)
                idx += 1
        # train dataset size
        self.data_size = len(self.id_list)  # 35532 for cvusCVM-Net-I_syd_originala # length of train data
        # To append valid neighbor id's of every row in the train dataset

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

        # This is to load the sat dataset
        self.__cur_sat_id = 0  # for training
        self.id_sat_list = []
        self.id_sat_idx_list = []
        self.pano_sat_id_list = []
        with open(self.train_sat_list, 'r') as file:
            idx = 0
            li = 0
            for line in file:
                # print(line)
                li += 1
                # print('li ,',li)
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_sat_list.append([data[0], data[1], pano_id])
                self.id_sat_idx_list.append(idx)
                self.pano_sat_id_list.append(pano_id)
                idx += 1
        self.sat_data_size = len(self.id_sat_list)  # 8884 for cvusa
        print('InputData::__init__: load', self.train_sat_list, ' data_size =', self.sat_data_size)

    # for sat
    def next_sat_batch(self, batch_size):
            print('self.curr_sat_id ', self.__cur_sat_id)
            # self.__cur_id is the starting pointer of a given batch
            if self.__cur_sat_id == 0:
                print('starting sat epoch')
                # do this only once
                c = list(zip(self.pano_sat_id_list, self.id_sat_list))
                random.shuffle(c)
                self.pano_sat_id_list, self.id_sat_list = zip(*c)
                del(c)
                # checking for valid sequence
                for bn in range (self.sat_data_size):
                    batch_indices = []
                    repeats = []
                    end_batch_ptr = (bn+1)*batch_size

                    for elem in range (batch_size):
                        index_is = bn*batch_size + elem
                        if (index_is >= self.sat_data_size):
                            break
                        # print(bn*batch_size , elem)
                        # print(index_is, self.data_size, len(self.pano_id_list))
                        c_val = self.pano_sat_id_list[index_is]
                        if (c_val in batch_indices):
                            repeats.append([c_val, index_is])
                        else:
                            batch_indices.append(c_val)

                    for r in repeats:
                        c_val, index_is = r[0], r[1]
                        chosen_val = random.choice(self.pano_sat_id_list[end_batch_ptr:])
                        chosen_ind = chosen_val.index(chosen_val)
                        self.pano_sat_id_list = list(self.pano_sat_id_list)
                        self.id_sat_list = list(self.id_sat_list)

                        if (chosen_val not in batch_indices):
                            # swap
                            self.pano_sat_id_list[index_is], self.pano_sat_id_list[chosen_ind] = self.pano_sat_id_list[chosen_ind], self.pano_sat_id_list[index_is]
                            self.id_sat_list[index_is], self.id_sat_list[chosen_ind] = self.id_sat_list[chosen_ind], self.id_sat_list[index_is]
                        # choose random row from beyond end_batch_ptr
                    # print ('past this')
                    # print(batch_indices)
                    if (index_is >= self.sat_data_size):
                        break
            # print(self.id_sat_list[:20])

            if self.__cur_sat_id + batch_size + 2 >= self.sat_data_size:
                self.__cur_sat_id = 0
                return None, None

            batch_sat_anchor = np.zeros([batch_size * batch_size, 512, 512, 3], dtype=np.float32)
            batch_sat_positive = np.zeros([batch_size * batch_size, 512, 512, 3], dtype=np.float32)

            i = 0
            batch_idx = 0
            # only loop for one batch
            while True:
                if batch_idx >= batch_size or self.__cur_sat_id + i >= self.sat_data_size:
                    break

                img_idx = self.id_sat_idx_list[self.__cur_sat_id + i]
                i += 1

                # satellite
                img_path = self.img_root + '/' + self.id_sat_list[img_idx][0]
                img = cv2.imread(img_path)
                # print("img anchor", img_path)
                # print(img_path)
                if img is None or img.shape[0] != img.shape[1]:
                    print(img_path)
                    print(
                        'InputData::next_pair_batch: read fail: %s, %d, ' % (
                            self.img_root + '/' + self.id_sat_list[img_idx][0], i),
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
                batch_sat_anchor[batch_idx, :, :, :] = img

                # satellite
                img_path = self.img_root + '/' + self.id_sat_list[img_idx][1]
                # print('img positi',img_path)
                img = cv2.imread(img_path)
                # print('img_p', img)
                # print("img")
                # print(img_path)
                if img is None or img.shape[0] != img.shape[1]:
                    print(img_path)
                    print(
                        'InputData::next_pair_batch: read fail: %s, %d, ' % (
                            self.img_root + '/' + self.id_sat_list[img_idx][1], i),
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
                batch_sat_positive[batch_idx, :, :, :] = img

                batch_idx += 1
            # move the starting pointer to the next batch
            self.__cur_sat_id += i
            # print(batch_sat_anchor)
            # print(batch_sat_positive)

            print('returning')

            return batch_sat_anchor, batch_sat_positive

    # this is used while training
    def next_pair_batch(self, batch_size):
            # self.__cur_id is the starting pointer of a given batch
            if self.__cur_id == 0:
                print('starting a new epoch')
                # do this only once
                c = list(zip(self.pano_id_list, self.id_list))
                random.shuffle(c)
                self.pano_id_list, self.id_list = zip(*c)
                del(c)
                # checking for valid sequence 
                for bn in range (self.data_size):
                    batch_indices = []
                    repeats = []
                    end_batch_ptr = (bn+1)*batch_size

                    for elem in range (batch_size):
                        index_is = bn*batch_size + elem
                        if (index_is >= self.data_size):
                            break
                        # print(bn*batch_size , elem)
                        # print(index_is, self.data_size, len(self.pano_id_list))
                        c_val = self.pano_id_list[index_is]
                        if (c_val in batch_indices):
                            repeats.append([c_val, index_is])
                        else: 
                            batch_indices.append(c_val)

                    for r in repeats:
                        c_val, index_is = r[0], r[1]

                        chosen_val = random.choice(self.pano_id_list[end_batch_ptr:])
                        chosen_ind = chosen_val.index(chosen_val)
                        self.pano_id_list = list(self.pano_id_list)
                        self.id_list = list(self.id_list)

                        if (chosen_val not in batch_indices):
                            # swap 
                            self.pano_id_list[index_is], self.pano_id_list[chosen_ind] = self.pano_id_list[chosen_ind], self.pano_id_list[index_is]
                            self.id_list[index_is], self.id_list[chosen_ind] = self.id_list[chosen_ind], self.id_list[index_is]
                        # choose random row from beyond end_batch_ptr 
                    # print ('past this')
                    # print(batch_indices)
                    if (index_is >= self.data_size):
                        break

            if self.__cur_id + batch_size + 2 >= self.data_size:
                self.__cur_id = 0
                return None, None

            batch_sat = np.zeros([batch_size * batch_size, 512, 512, 3], dtype=np.float32)
            batch_grd = np.zeros([batch_size * batch_size, 224, 1232, 3], dtype=np.float32)
            i = 0
            batch_idx = 0
            # only loop for one batch
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

                batch_idx += 1
            # move the starting pointer to the next batch
            self.__cur_id += i
            print('returning')
            return batch_sat, batch_grd

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
            print('hei')

            print(ground_view_img)
            try:
                img = img.astype(np.float32)
            except:
                print(img)


            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_grd[i, :, :, :] = img

        self.__cur_test_id += batch_size
        print('returning input')
        return batch_sat, batch_grd

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