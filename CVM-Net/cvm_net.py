from VGG import VGG16
import loupe as lp
from siamese_fc import Siamese_FC
from transnet_v2 import TransNet

import tensorflow as tf

# def cvm_net_sat_I(x_sat, x_grd, keep_prob, trainable):
#     with tf.device('/gpu:1'):
#         # local descriptors / local feature extraction; use only the convolution layers (until conv_5) of vgg for this
#         vgg_sat_p = VGG16()
#         # output of conv_5 layer; the input dimension and output dimension of conv2 layer (512 x 512)
#         sat_p_local = vgg_sat_p.VGG16_conv(x_grd, keep_prob, False, 'VGG_sat')
#         with tf.variable_scope('netvlad_sat', reuse=tf.AUTO_REUSE):
#             # embed netvlad on each cnn branch to get global descriptor

#             # constructor!
#             # feature size is the depth of the final convolution layer
#             # grd local
#             netvlad_sat_p = lp.NetVLAD(feature_size=512, max_samples=tf.shape(sat_p_local)[1] * tf.shape(sat_p_local)[2],
#                                      cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
#                                      is_training=trainable)
#             # forward prop; give input of conv_5 layer
#             # output is the vlad vector (K*D) x 1 (64*512) x 1
#             sat_p_vlad = netvlad_sat_p.forward(sat_p_local)

#         vgg_sat_a = VGG16()
#         # local satellite descriptors ;
#         sat_a_local = vgg_sat_a.VGG16_conv(x_sat, keep_prob, False, 'VGG_sat')
#         with tf.variable_scope('netvlad_sat', reuse=tf.AUTO_REUSE):
#             # shape of sat_local is batch_sz x 14 x 14 x 512 (i guess)
#             # so max samples is 14*14 = 196? is this equal to N?
#             # is cluster size equal to K or N?
#             netvlad_sat_a = lp.NetVLAD(feature_size=512, max_samples=tf.shape(sat_a_local)[1] * tf.shape(sat_a_local)[2],
#                                      cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
#                                      is_training=trainable)
#             sat_a_vlad = netvlad_sat_a.forward(sat_a_local)

#     with tf.device('/gpu:0'):
#         fc = Siamese_FC()
#         sat_a_global, sat_p_global = fc.siamese_fc(sat_a_vlad, sat_p_vlad, trainable, 'dim_reduction')

#     return sat_a_global, sat_p_global

# satellite input, ground input, dropout keep probability, (bool) should the parameters train?
# def cvm_net_I(x_sat, x_grd, keep_prob, trainable):
#     with tf.device('/gpu:1'):
#         # local descriptors / local feature extraction; use only the convolution layers (until conv_5) of vgg for this 
#         vgg_grd = VGG16()
#         # output of conv_5 layer; the input dimension and output dimension of conv2 layer (512 x 512)
#         grd_local = vgg_grd.VGG16_conv(x_grd, keep_prob, False, 'VGG_grd')
#         with tf.variable_scope('netvlad_grd', reuse=tf.AUTO_REUSE):
#             # embed netvlad on each cnn branch to get global descriptor

#             # constructor!
#             # feature size is the depth of the final convolution layer
#             # grd local 
#             netvlad_grd = lp.NetVLAD(feature_size=512, max_samples=tf.shape(grd_local)[1] * tf.shape(grd_local)[2],
#                                      cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
#                                      is_training=trainable)
#             # forward prop; give input of conv_5 layer
#             # output is the vlad vector (K*D) x 1 (64*512) x 1
#             grd_vlad = netvlad_grd.forward(grd_local)

#         vgg_sat = VGG16()
#         # local satellite descriptors ; 
#         sat_local = vgg_sat.VGG16_conv(x_sat, keep_prob, False, 'VGG_sat')
#         with tf.variable_scope('netvlad_sat', reuse=tf.AUTO_REUSE):
#             # shape of sat_local is batch_sz x 14 x 14 x 512 (i guess)
#             # so max samples is 14*14 = 196? is this equal to N?
#             # is cluster size equal to K or N?
#             netvlad_sat = lp.NetVLAD(feature_size=512, max_samples=tf.shape(sat_local)[1] * tf.shape(sat_local)[2],
#                                      cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
#                                      is_training=trainable)
#             sat_vlad = netvlad_sat.forward(sat_local)

#     with tf.device('/gpu:0'):
#         fc = Siamese_FC()
#         sat_global, grd_global = fc.siamese_fc(sat_vlad, grd_vlad, trainable, 'dim_reduction')

#     return sat_global, grd_global

def cvm_net_I(x_sat, x_grd, coords_geo, keep_prob, trainable):
    with tf.device('/gpu:1'):
        # local descriptors / local feature extraction; use only the convolution layers (until conv_5) of vgg for this 
        vgg_grd = VGG16()
        # output of conv_5 layer; the input dimension and output dimension of conv2 layer (512 x 512)
        grd_local = vgg_grd.VGG16_conv(x_grd, keep_prob, False, 'VGG_grd')
        with tf.variable_scope('netvlad_grd', reuse=tf.AUTO_REUSE):
            # embed netvlad on each cnn branch to get global descriptor

            # constructor!
            # feature size is the depth of the final convolution layer
            # grd local 
            netvlad_grd = lp.NetVLAD(feature_size=512, max_samples=tf.shape(grd_local)[1] * tf.shape(grd_local)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=trainable)
            # forward prop; give input of conv_5 layer
            # output is the batch*(vlad vector) i.e 12 * [(K*D) x 1] (64*512) x 1
            grd_vlad = netvlad_grd.forward(grd_local)

        vgg_sat = VGG16()
        # local satellite descriptors ; 
        sat_local = vgg_sat.VGG16_conv(x_sat, keep_prob, False, 'VGG_sat')
        with tf.variable_scope('netvlad_sat', reuse=tf.AUTO_REUSE):
            # shape of sat_local is batch_sz x 14 x 14 x 512 (i guess)
            # so max samples is 14*14 = 196? is this equal to N?
            # is cluster size equal to K or N?
            netvlad_sat = lp.NetVLAD(feature_size=512, max_samples=tf.shape(sat_local)[1] * tf.shape(sat_local)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=trainable)
            sat_vlad = netvlad_sat.forward(sat_local)

    with tf.device('/gpu:0'):
        fc = Siamese_FC()
        sat_global, grd_global = fc.siamese_fc(sat_vlad, grd_vlad, trainable, 'dim_reduction')
        # grd_global = fc.new_fc_layer(sat_global, coords_geo)
    return sat_global, grd_global


def cvm_net_II(x_sat, x_grd, keep_prob, trainable):
    with tf.device('/gpu:1'):
        vgg_grd = VGG16()
        grd_local = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')

        vgg_sat = VGG16()
        sat_local = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')

        transnet = TransNet()
        trans_sat, trans_grd = transnet.transform(sat_local, grd_local, keep_prob, trainable,
                                                  'transformation')

        with tf.variable_scope('netvlad') as scope:
            netvlad_sat = lp.NetVLAD(feature_size=512, max_samples=tf.shape(trans_sat)[1] * tf.shape(trans_sat)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=trainable)
            sat_global = netvlad_sat.forward(trans_sat, True)

            scope.reuse_variables()

            netvlad_grd = lp.NetVLAD(feature_size=512, max_samples=tf.shape(trans_grd)[1] * tf.shape(trans_grd)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=trainable)
            grd_global = netvlad_grd.forward(trans_grd, True)

    return sat_global, grd_global

