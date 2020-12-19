from typing import List, Any, Union

from cvm_net import cvm_net_I, cvm_net_II
from input_data import InputData
from input_data import ValidateData
from input_data import TestData
from matplotlib import pyplot as plt
import tensorflow as tf
import scipy.io as sio
import numpy as np
import sys
from fastkml import kml
np.set_printoptions(threshold=sys.maxsize)
import os
import io
import cv2
from operator import itemgetter
from scipy.io import loadmat
from os import path

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
# network_type = 'CVM-NET-II'
network_type = 'CVM-NET-I'
network_name = 'CVM-Net-I'
batch_size = 12
# batch_size = 4
is_training = True
loss_weight = 10.0
number_of_epoch = 120

learning_rate_val = 0.00001
keep_prob_val = 0.8  # neuron keep probability for dropout

# this validate function is if you pass a single image by image index
# Use this while doing testing for one image
def validate(grd_descriptor, sat_descriptor_file, k, gnd_gt=0):
    descriptors = loadmat(sat_descriptor_file)
    sat_descriptor = descriptors['sat_global_descriptor']
    total_descriptors = sat_descriptor.shape[0]
    # print(np.shape(grd_descriptor))
    # print(grd_descriptor)
    # grd_descriptor = np.tile(grd_descriptor,(total_descriptors, 1))

    # cosine distance to calculate similarity
    # compare each ground descriptor with all the
    # satellite descriptors and assess (Cosine) similarity
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    # print('dist_array ',dist_array)
    # it is better you calculate top 10 per cent images for a smaller dataset like ours
    topk_percent = int(dist_array.shape[0] * (k / 100.0)) + 1
    # sort dist_array in ascending order and consider top k values and their corresponding index
    # topk_matches = sorted(dist_array, key = lambda x:float(x))
    topk_matches = [i[0] for i in sorted(enumerate(dist_array), key=lambda x:x[1])]
    # gt_position = np.where(topk_matches == gnd_gt) # gnd_gt is the ground image index
    # these are the closest image indices
    topk_matches = topk_matches[:topk_percent]
    # print ('topk_matches ', topk_matches)

    # gt_position = topk_matches.index(gnd_gt)
    # print('gt was found at position ', gt_position+1)
    # print('the top k percent is ', topk_percent)
    return topk_matches

def compute_loss(sat_global, grd_global, batch_hard_count=0):
    '''
    Compute the weighted soft-margin triplet loss
    :param sat_global: the satellite image global descriptor
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the loss
    '''
    with tf.name_scope('weighted_soft_margin_triplet_loss'):
        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
        pos_dist = tf.diag_part(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
            top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = tf.reduce_mean(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
            top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = tf.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss


def train(start_epoch=1):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''

    # import data (get the train and validation data) in the format
    # satellite filename, streetview filename, pano_id
    # its job is to just create a python version of the list that's already
    # there in the test file

    ### NOTE: JUST CHANGE THE TEST FILE ###
    input_data = InputData()

    # define placeholders to feed actual training examples
    # size of the actual images sat-ellite and ground
    # satellite (512, 512) image shape
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
    # ground (224, 1232) image shape
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x')
    keep_prob = tf.placeholder(tf.float32)  # dropout
    learning_rate = tf.placeholder(tf.float32)

    # just BUILDING MODEL, satellite and ground image will be given later
    if network_type == 'CVM-NET-I':
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
    elif network_type == 'CVM-NET-II':
        sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, is_training)
    else:
        print ('CONFIG ERROR: wrong network type, only CVM-NET-I and CVM-NET-II are valid')

    # define loss
    loss = compute_loss(sat_global, grd_global, 0)

    # set training
    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    plt.ion()
    plt.xlabel('epochs')
    plt.ylabel('loss/accuracy')
    # plt.show()
    p = []
    p_val = []
    p_acc = []
    p_acc_ = []
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        print('load model...')

        ### dont uncomment ###
        # load_model_path = '../Model/' + network_type + '/' + str(start_epoch - 1) + '/model.ckpt'
        # saver.restore(sess, load_model_path)
        # print("   Model loaded from: %s" % load_model_path)
        # print('load model...FINISHED')

        os.chdir('../Model/')
        cwd = os.getcwd()

        model_name = 'syd_orig_90'
        # which_epoch = 77

        load_model_path = cwd + '/' + network_name + '/' + network_name + '_model'
        if (start_epoch > 1):
            load_model_path = cwd + '/' + network_name + '/' + network_name + '_' + model_name + '/' + network_type + '/' + str(
                start_epoch)

        saver = tf.train.import_meta_graph(load_model_path + "/model.ckpt.meta")
        load_model_path += '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        print('training...')

        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            train_loss = []
            val_loss = []
            while True:
                # train
                batch_sat, batch_grd = input_data.next_pair_batch(batch_size)
                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat_x: batch_sat, grd_x: batch_grd,
                             learning_rate: learning_rate_val, keep_prob: keep_prob_val}

                _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                train_loss.append(loss_val)
                # print("run model")
                # print('global %d, epoch %d, iter %d: loss : %.4f' %
                #       (global_step_val, epoch, iter, loss_val))
                train_loss.append(loss_val)
                iter += 1

            plt.legend()
            p += [np.mean(train_loss)]
            plt.plot(p, 'b-')
            plt.pause(0.05)

            # ---------------------- validation ----------------------
            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()
            sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
            grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
            val_i = 0
            while True:
                print('      progress %d' % val_i)
                # get the sat and grd batch; this is just the input images
                batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break  # break once all batches are over
                # create a dictionary
                feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}

                # this dictionary stores all the global descriptors
                sat_global_val, grd_global_val = \
                    sess.run([sat_global, grd_global], feed_dict=feed_dict)
                # print('sat_global_val ', sat_global_val)

                val_loss.append(sess.run(loss, feed_dict=feed_dict))

                sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                val_i += sat_global_val.shape[0]  # is this 64*512?

            # print('val_loss ', val_loss)
            p_val += [np.mean(val_loss)]
            plt.plot(p_val, 'r-')
            plt.pause(0.05)

            print('   compute accuracy')
            val_accuracy, val_accuracy_ = validate(grd_global_descriptor, sat_global_descriptor)
            p_acc += [val_accuracy]
            p_acc_ += [val_accuracy_]
            plt.plot(p_acc, 'k-')
            plt.pause(0.05)
            plt.plot(p_acc_, 'g-')
            plt.pause(0.05)

            # with open('../Result/' + str(network_type) + '_accuracy.txt', 'a') as file:
            #     file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')
            print('   %d: accuracy_10percent = %.1f%%' % (epoch, val_accuracy * 100.0))
            print('   %d: accuracy_1percent = %.1f%%' % (epoch, val_accuracy_ * 100.0))
            cwd = os.getcwd()

            if (epoch == 119):
                print('validate all...')
                print('   compute global descriptors')
                input_data.reset_scan()
                sat_global_descriptor = np.zeros([input_data.get_tt_dataset_size(), 4096])
                grd_global_descriptor = np.zeros([input_data.get_tt_dataset_size(), 4096])
                val_i = 0
                val_loss = []
                while True:
                    print('      progress %d' % val_i)
                    # get the sat and grd batch; this is just the input images
                    batch_sat, batch_grd = input_data.next_tt_scan(batch_size)
                    if batch_sat is None:
                        break  # break once all batches are over
                    # create a dictionary
                    feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}

                    # this dictionary stores all the global descriptors
                    sat_global_val, grd_global_val = \
                        sess.run([sat_global, grd_global], feed_dict=feed_dict)
                    # print('sat_global_val ', sat_global_val)

                    val_loss.append(sess.run(loss, feed_dict=feed_dict))

                    sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                    grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                    val_i += sat_global_val.shape[0]  # is this 64*512?

                print('   compute accuracy')
                total_val_accuracy, total_val_accuracy_ = validate(grd_global_descriptor, sat_global_descriptor)

                # with open('../Result/' + str(network_type) + '_accuracy.txt', 'a') as file:
                #     file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')
                print('   %d: accuracy_10percent = %.1f%%' % (epoch, val_accuracy * 100.0))
                print('   %d: accuracy_1percent = %.1f%%' % (epoch, val_accuracy_ * 100.0))
                print('   %d: accuracy_tt_10percent = %.1f%%' % (epoch, total_val_accuracy * 100.0))
                print('   %d: accuracy_tt_1percent = %.1f%%' % (epoch, total_val_accuracy_ * 100.0))
                cwd = os.getcwd()

                os.chdir('../Model/CVM-Net-I/CVM-Net-I_'+model_name+'/')
                cwd = os.getcwd()
                os.chdir('../../../CVM-Net/')

                model_dir = cwd + '/' + network_type + '/' + str(epoch) + '/'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                    # if (epoch > 70 or epoch % 5 == 0):
                    save_path = saver.save(sess, model_dir + 'model.ckpt')
                    # sio.savemat(model_dir + 'np_vector_CVM_Net.mat', {'sat_global_descriptor': sat_global_descriptor,
                    #                                                   'grd_global_descriptor': grd_global_descriptor})
                    print("Model saved in file: %s" % save_path)

                # model_dir = cwd + '/' + network_type + '/' + str(epoch) + '/'
                model_dir = cwd + '/' + network_type
                print("save dir ", model_dir)
                sio.savemat(model_dir + '/all_descriptors.mat', {'sat_global_descriptor': sat_global_descriptor,
                                                                  'grd_global_descriptor': grd_global_descriptor})

class TestImage:
    def __init__(self, k_, image):

        print("Checking top %d percent neighbors for image" % k_)
        # append points
        # collect the points which have correspondences

        self.image_path = image
        self.k = k_

        cwd = os.getcwd()

        ind_file = open('/scratch1/surabhi/crossview_localisation/src/CVM-Net/index.txt', "r")

        # Reading from file
        Content = ind_file.read()
        CoList = Content.split("\n")
        # my_index = []

        points_ = []
        with open('/scratch1/surabhi/crossview_localisation/src/CVM-Net/New_York.csv', 'r') as file:
            for line in file:
                data = line.split(',')
                points_.append([data[1], data[0]])
        self.points = []
        for co in CoList:
            if not co == '':
                co_ = int(co)
                # print(type(co_))
                curre = points_[co_]
                self.points.append([curre[0], curre[1]])

        # with io.open('/scratch1/surabhi/crossview_localisation/src/CVM-Net/new_path.kml', 'rt', encoding="utf-8") as myfile:
        #     doc = myfile.read().encode()
        # k = kml.KML()
        # k.from_string(doc)
        # features = list(k.features())
        # f2 = list(features[0].features())
        # data = f2[0].geometry
        #
        # self.points = []
        # for p in data.coords:
        #     self.points.append([p[1], p[0]])

    def find_knn(self):
        tf.reset_default_graph()
        # where is the trained model stored?
        model_name = 'nyc/'
        # which trained epoch model do you want to use?
        which_epoch = 119
        # name of the file where satellite descriptors are stored
        sat_descriptor_file = 'all_nyc_descriptors.mat'
        # what is the name of the folder where you will find the images?
        Data_folder = 'nyc'

        # import data (get the test data) in the format,
        # uncomment the other TestData class in input_data.py if you are using this
        input_data = TestData(Data_folder, self.image_path)
        # define placeholders to feed actual training examples
        # size of the actual images satellite and ground
        # satellite (512, 512) image shape
        sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
        # ground (224, 1232) image shape
        grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x')
        keep_prob = tf.placeholder(tf.float32)  # dropout

        # just BUILDING MODEL, satellite and ground image will be given later
        if network_type == 'CVM-NET-I':
            sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
        elif network_type == 'CVM-NET-II':
            sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, is_training)
        else:
            print ('CONFIG ERROR: wrong network type, only CVM-NET-I and CVM-NET-II are valid')

        # This variable downgrades the accuracy, but why?
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        # run model
        print('run model...')
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())

            print('load model...')

            os.chdir('../Data/')
            image_store = os.getcwd()

            os.chdir('../Model/')
            cwd = os.getcwd()

            load_model_path = cwd + '/' + network_name + '/' + network_name + '_' + model_name + '/' + network_type + '/' + str(which_epoch)
            sat_descriptor_file = '/scratch1/crossview2/descriptors/' + sat_descriptor_file

            saver = tf.train.import_meta_graph(load_model_path + "/model.ckpt.meta")
            load_model_path += '/model.ckpt'
            saver.restore(sess, load_model_path)
            print("   Model loaded from: %s" % load_model_path)
            print('load model...FINISHED')

            # ---------------------- Testing ----------------------
            print('Test...')
            print('compute global descriptors')
            input_data.reset_scan()
            sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
            grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
            # grd_global_descriptor = np.zeros([1, 4096])
            val_i = 0
            while True:
                print('progress %d' % val_i)
                # get the sat and grd batch; this is just the input images
                batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break  # break once all batches are over
                # create a dictionary
                feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}

                # this dictionary stores all the global descriptors
                sat_global_val, grd_global_val = \
                    sess.run([sat_global, grd_global], feed_dict=feed_dict)

                sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                val_i += sat_global_val.shape[0]  # is this 64*512?

            print('compute accuracy')
            topk_images = validate(grd_global_descriptor, sat_descriptor_file, self.k)
            # print(np.shape(topk_images))

            gnd_image_paths = []
            sat_image_paths = []
            gps_coordinates = []
            for im in topk_images:
                gps_coordinates.append(self.points[im])
                gnd_image_path = image_store + '/streetview/' + str(im) + '.jpg'
                sat_image_path = image_store + '/satellite/' + str(im) + '.jpg'
                gnd_image_paths.append(gnd_image_path)
                sat_image_paths.append(sat_image_path)
            # print(np.shape(gnd_image_paths))
            # print(np.shape(sat_image_paths))
            return sat_image_paths, gnd_image_paths, gps_coordinates

# get descriptors of all the ground and satellite images in the train set
def get_descriptors():

    model_name = 'nyc'
    which_epoch = 119
    Data_folder = 'nyc'
    # get descriptors of images mentioned in this file
    # Test_file = 'train-19zl.csv'
    Test_file = 'train-test.csv'
    save_file = 'all_nyc_descriptors.mat'

    input_data = ValidateData(Data_folder, Test_file)

    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x')
    keep_prob = tf.placeholder(tf.float32)

    if network_type == 'CVM-NET-I':
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
    elif network_type == 'CVM-NET-II':
        sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, is_training)
    else:
        print ('CONFIG ERROR: wrong network type, only CVM-NET-I and CVM-NET-II are valid')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')
        os.chdir('../Model/')

        cwd = os.getcwd()
        # load the interleaved sydney dataset
        load_model_path = cwd + '/' + network_name + '/' + network_name + '_'+ model_name + '/' + network_type + '/' + str(which_epoch)
        # print(load_model_path)
        saver = tf.train.import_meta_graph(load_model_path + "/model.ckpt.meta")
        load_model_path += '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        print('testing...')

        print('   compute global descriptors')
        input_data.reset_scan()
        sat_global_descriptor = np.zeros(
            [input_data.get_test_dataset_size(), 4096])
        grd_global_descriptor = np.zeros(
            [input_data.get_test_dataset_size(), 4096])
        val_i = 0
        # this is for train
        while True:
            print('progress %d' % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break  # break when all the batches are evaluated
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            # works fine until here
            # forward pass
            sat_global_val, grd_global_val = \
                sess.run([sat_global, grd_global], feed_dict=feed_dict)  # feed in the batch input here

            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        cwd = os.getcwd()
        os.chdir('../Model/CVM-Net-I/CVM-Net-I_'+ model_name +'/')
        cwd = os.getcwd()
        os.chdir('../../../CVM-Net/')

        model_dir = cwd + '/' + network_type + '/'
        # print('compute accuracy')
        # # This would be the train and test accuracy
        # val_accuracy, val_accuracy_ = validate(grd_global_descriptor, sat_global_descriptor)
        # print("10_percent ",val_accuracy)
        # print("1_percent ", val_accuracy_)
        sio.savemat(model_dir + save_file,
                    {'sat_global_descriptor': sat_global_descriptor, 'grd_global_descriptor': grd_global_descriptor})

if __name__ == '__main__':
    # test()
    train()
    # get_descriptors()