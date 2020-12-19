
from cvm_net import cvm_net_I, cvm_net_II
from input_data import InputData
from input_data import TestData

import tensorflow as tf
import numpy as np
import os
import cv2
from operator import itemgetter
from os import path

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
# network_type = 'CVM-NET-II'
network_type = 'CVM-NET-I'
network_name = 'CVM-Net-I'

batch_size = 3 # (12) 4 sets of triplets! query, close satellite, far satellite
is_training = True
loss_weight = 10.0
number_of_epoch = 5 # half the lr in every 5 epochs

learning_rate_val = 1e-5
keep_prob_val = 0.8 # neuron keep probability for dropout
# -------------------------------------------------------- #

# this gives the accuracy of the network i.e how similar is the global 
# satellite and global ground descriptors
def validate(grd_descriptor, sat_descriptor):
    print("validating now..")
    accuracy = 0.0
    data_amount = 0.0
    # cosine distance to calculate similarity
    # compare each ground descriptor with all the 
    # satellite descriptors and assess (Cosine) similarity 
    # (123 x 123)
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    percent = 10
    # it is better you calculate top 10 per cent images for a smaller dataset like ours
    top1_percent = int(dist_array.shape[0] * (percent/100.0)) + 1

    # dist_array.shape[0] - number of samples
    for i in range(dist_array.shape[0]):
	# diagonal elements will ideally have max val amongst all the other elements 
	# because the similarity between the corresponding ground and satellite 
	# images should be the highest (or in the case of the formula the lowest). 
	# so if the similarity is lower than this for any other image, then that 
	# image is more similar.  

        gt_dist = dist_array[i, i] 
	# Amongst all the satellite images (analyzing column-wise), what all elements 
	# are less than the diagonal value   
        
        
        my_list = dist_array[:,i]
	#sorted(range(len(my_list)),key=my_list.__getitem__)
        indices, list_sorted = zip(*sorted(enumerate(my_list), key=itemgetter(1)))

        #print(list(indices))
        #print(list(list_sorted))
        #print(dist_array[:, i] < gt_dist)
        #print(np.where(dist_array[:, i] < gt_dist))

        # sort index array according to dist array.. 
        prediction = np.sum(dist_array[:, i] < gt_dist)
        #print(prediction)
        # if accuracy is less that means there are way too many other
        # (not corresponding) satellite images that are "similar" to the query image
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0


    accuracy /= data_amount

    return accuracy

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
    input_data = InputData()

    # define placeholders to feed actual training examples
    # size of the actual images sat-ellite and ground
    #satellite (512, 512) image shape
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x') 
    #ground (224, 1232) image shape
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x') 
    keep_prob = tf.placeholder(tf.float32) #dropout
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
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        print('load model...')
        # load_model_path = '../Model/' + network_type + '/' + str(start_epoch - 1) + '/model.ckpt'
        # saver.restore(sess, load_model_path)
        # print("   Model loaded from: %s" % load_model_path)
        # print('load model...FINISHED')

        os.chdir('../../Model/')
   
        cwd=os.getcwd()
        load_model_path = cwd + '/' + network_name + '/' + network_name + '_model'
        print(load_model_path)
        saver=tf.train.import_meta_graph(load_model_path+"/model.ckpt.meta")
        print('????????')
        load_model_path += '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')
        import tensorflow.contrib.slim as slim
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        print('training...')

        # Train
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                # train
                batch_sat, batch_grd = input_data.next_pair_batch(batch_size)
                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat_x: batch_sat, grd_x: batch_grd,
                             learning_rate: learning_rate_val, keep_prob: keep_prob_val}
                print("run model")
                if iter % 20 == 0:
                    print('running {}'.format(iter))
                    _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                    print('global %d, epoch %d, iter %d: loss : %.4f' %
                          (global_step_val, epoch, iter, loss_val))
                else:
                    print("running")
                    sess.run(train_step, feed_dict=feed_dict)
                print("ran once?")
                iter += 1
            
        #     # ---------------------- validation ----------------------
        #     print('validate...')
        #     print('   compute global descriptors')
        #     input_data.reset_scan()
        #     sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        #     grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        #     val_i = 0
        #     while True:
        #         print('      progress %d' % val_i)
        #         # get the sat and grd batch; this is just the input images
        #         batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
        #         if batch_sat is None:
        #             break # break once all batches are over
        #         # create a dictionary
        #         feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
                
        #         # this dictionary stores all the global descriptors
        #         sat_global_val, grd_global_val = \
        #             sess.run([sat_global, grd_global], feed_dict=feed_dict)
        #             # sat_global and grd_global is the vlad output layer value 
        #             # this is the forward pass 

        #         sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
        #         grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
        #         val_i += sat_global_val.shape[0] # is this 64*512? 

        #     print('   compute accuracy')
        #     val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
        #     with open('../Result/' + str(network_type) + '_accuracy.txt', 'a') as file:
        #         file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')
        #     print('   %d: accuracy = %.1f%%' % (epoch, val_accuracy*100.0))

        #     model_dir = '../Model/' + network_type + '/' + str(epoch) + '/'
        #     if not os.path.exists(model_dir):
        #         os.makedirs(model_dir)
        #     save_path = saver.save(sess, model_dir + 'model.ckpt')
        #     print("Model saved in file: %s" % save_path)
            # ---------------------------------------------------------
def test():
    
    input_data = TestData()
    print("test_model")
    
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x') 
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x') 
    keep_prob = tf.placeholder(tf.float32) #dropout
    learning_rate = tf.placeholder(tf.float32) 

    # just BUILDING MODEL, satellite and ground image will be given later
    if network_type == 'CVM-NET-I':
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
    
    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        print('load model...')
        os.chdir('../../Model/')
   
        cwd=os.getcwd()
        load_model_path = cwd + '/' + network_name + '/' + network_name + '_model'
        print(load_model_path)
        saver=tf.train.import_meta_graph(load_model_path+"/model.ckpt.meta")
        load_model_path += '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')
        import tensorflow.contrib.slim as slim
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        print('testing...')

        print('   compute global descriptors')
        input_data.reset_scan()
        sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        val_i = 0

        while True:
            print('progress %d' % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break # break when all the batches are evaluated
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            # works fine until here
            # forward pass
            sat_global_val, grd_global_val = \
                sess.run([sat_global, grd_global], feed_dict=feed_dict)  # feed in the batch input here

            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        print('compute accuracy')
        val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
        print(' accuracy = %.1f%%' % (val_accuracy*100.0))

def test_query(query_image):
    
    input_data = TestData()
    print("test_model")
    
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x') 
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x') 
    keep_prob = tf.placeholder(tf.float32) #dropout
    learning_rate = tf.placeholder(tf.float32) 

    # just BUILDING MODEL, satellite and ground image will be given later
    if network_type == 'CVM-NET-I':
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, is_training)
    
    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        print('load model...')
        os.chdir('../Model/')
        cwd=os.getcwd()
        os.chdir('../CVM-Net')

        load_model_path = cwd + '/' + network_name + '/' + network_name + '_model'
        saver=tf.train.import_meta_graph(load_model_path+"/model.ckpt.meta")
        load_model_path += '/model.ckpt'
        saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')
        # print model 
        # import tensorflow.contrib.slim as slim
        # model_vars = tf.trainable_variables()
        # slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        print('testing...')

        print('   compute global descriptors')
        input_data.reset_scan()
        sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 4096])
        val_i = 0
        # this is just pre processing step
        while True: 
            print('progress %d' % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break # break when all the batches are evaluated
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            # works fine until here
            # forward pass
            sat_global_val, grd_global_val = \
                sess.run([sat_global, grd_global], feed_dict=feed_dict)  # feed in the batch input here

            sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
            grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
            val_i += sat_global_val.shape[0]

        print('compute accuracy')
        # This is where you compare the each individual ground descriptor in the 
        # dataset 
        query_id = int((query_image.split('/')[-1]).split('.')[0])
        # dist array is 123 x 1
        dist_array = 2 - 2 * np.matmul(sat_global_descriptor, np.transpose(grd_global_descriptor[query_id,:]))
        k = 10
        # it is better you calculate top 10 per cent images for a smaller dataset like ours
        topk_percent = int(dist_array.shape[0] * (k/100.0)) + 1
        
	#sorted(range(len(my_list)),key=my_list.__getitem__)
        indices, list_sorted = zip(*sorted(enumerate(dist_array), key=itemgetter(1)))
        list_sorted = list(list_sorted)
        indices = list(indices)
        k_indices = indices[:k]
        k_vals = list_sorted[:k]
        # these are the top k image indices
        img_root = os.getcwd()
        os.chdir('../Data/Google_dataset/')
        img_root = os.getcwd()
        os.chdir('../../CVM-Net/')
        
        for i in range(len(k_indices)):
            img_index = k_indices[i]
            print(img_index)
            img_path = img_root + '/satellite/' + str(img_index) + '.jpg'
            #print('img_path ',img_path)
            c_img = cv2.imread(img_path)
            save_path = os.getcwd() + '/top_k/' + str(query_id) + '/' + str(img_index) + '.jpg'
            #print('save_path ', save_path)
            #cv2.imwrite(save_path, c_img)
            #cv2.imshow('top k', c_img)
            #cv2.waitKey(0)
        print('\n')
        # these are the cosine similarity values
        for i in range(len(k_indices)):
            print(k_vals[i])
        # See what is the ranking of the actual ground truth in top k predictions. 
        try:
            gt_pos = k_indices.index(query_id)
        except:
            gt_pos = -1
        print('gt_pos ', gt_pos)

if __name__ == '__main__':
      
    print("hello")

    #cwd=os.getcwd()
    #print(cwd)
    os.chdir('../Data/Google_dataset/')
    img_root=os.getcwd()
    print(img_root)
    # going back to where this file is located
    os.chdir('../../CVM-Net')
    img_index = 0
    query_image=img_root+'/streetview/'+str(img_index)+'.jpg'
    test_query(query_image)
