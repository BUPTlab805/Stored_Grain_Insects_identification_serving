# -*- coding: utf-8 -*-
# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Definition of 512 VGG-based SSD network.

This model was initially introduced in:
SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325

Two variants of the model are defined: the 300x300 and 512x512 models, the
latter obtaining a slightly better accuracy on Pascal VOC.

Usage:
    with slim.arg_scope(ssd_vgg.ssd_vgg()):
        outputs, end_points = ssd_vgg.ssd_vgg(inputs)
@@ssd_vgg
"""
import math
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common
from nets import ssd_vgg_300

slim = tf.contrib.slim


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])
#mhw add isFPN
IS_FPN=True

class SSDNet(object):
    """Implementation of the SSD VGG-based 512 network.

    The default features layers with 512x512 image input are:
      conv4 ==> 64 x 64
      conv7 ==> 32 x 32
      conv8 ==> 16 x 16
      conv9 ==> 8 x 8
      conv10 ==> 4 x 4
      conv11 ==> 2 x 2
      conv12 ==> 1 x 1
    The default image size used to train this network is 512x512.
    """
    default_params = SSDParams(
        img_shape=(512, 512),
        num_classes=7,
        no_annotation_label=21,
        #               0         1         2         3          4          5        6
        feat_layers=['block4', 'block7', 'block8'],
        feat_shapes=[(64, 64), (32, 32), (16, 16)],
        anchor_size_bounds=[0.10, 0.90],
        anchor_sizes=[(20.48, 51.2),
                      (51.2, 133.12),
                      (133.12, 215.04)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3]],
        anchor_steps=[8, 16, 32],
        anchor_offset=0.5,
        normalizations=[20, -1, -1,],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_vgg'):
        """Network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        # print "==========detected_bboxes==========begin=========="
        # print "predictions = {}".format(predictions)
        # print "localisations = {}".format(localisations)


        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        # print ""
        # print "rscore = {}".format(rscores)
        # print "rbboxes = {}".format(rbboxes)


        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # print ""
        # print "rscore = {}".format(rscores)
        # print "rbboxes = {}".format(rbboxes)
        # Apply NMS algorithm.


        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        # if clipping_bbox is not None:
        #     rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        # print ""
        # print "detected_bboxes rscores={}".format(rscores)
        # print "detected_bboxes rbboxes={}".format(rbboxes)
        # print "==========detected_bboxes==========end=========="


        return rscores, rbboxes

    def losses(self, logits_pest, b_gclasses_pest, logits, localisations,b_gclasses, b_glocalisations, b_gscores, b_gscores_pest,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits_pest, b_gclasses_pest, logits, localisations,b_gclasses, b_glocalisations, b_gscores, b_gscores_pest,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def layer_shape(layer):
    """Returns the dimensions of a 4D layer tensor.
    Args:
      layer: A 4-D Tensor of shape `[height, width, channels]`.
    Returns:
      Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if layer.get_shape().is_fully_defined():
        return layer.get_shape().as_list()
    else:
        static_shape = layer.get_shape().with_rank(4).as_list()
        dynamic_shape = tf.unstack(tf.shape(layer), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(512, 512)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (512 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * 0.04, img_size * 0.1]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        shape = l.get_shape().as_list()[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    '''
        layer_shape : feature map形状 exg: 64*64
        size 是当前层anchor的大小
        ratios 长宽比例
        step 特征图的缩放比例
        img_shape = (512, 512)
        feat_shape = (64, 64)
        sizes = (20.48, 51.2)
        ratios = [2, 0.5]
        step = 8
        offset = 0.5


    '''
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    '''
    生成feature的每个点的坐标 x[:,i] = i y[j,:]=j  i=[0:shape[0]]
    y = [[ 0  0  0 ...  0  0  0]
         [ 1  1  1 ...  1  1  1]
         [ 2  2  2 ...  2  2  2]
         ...
         [61 61 61 ... 61 61 61]
         [62 62 62 ... 62 62 62]
         [63 63 63 ... 63 63 63]],  
     x = [[ 0  1  2 ... 61 62 63]
         [ 0  1  2 ... 61 62 63]
         [ 0  1  2 ... 61 62 63]
         ...
         [ 0  1  2 ... 61 62 63]
         [ 0  1  2 ... 61 62 63]
         [ 0  1  2 ... 61 62 63]]
    '''
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]

    '''
    下面四行是生成anchor的中心坐标
    其中 y.astype(dtype) + offset 是为了见坐标移到anchor的中心点 现在的坐标还是feature上进行的
    step代表了当前feature与原图的缩放比例 例如当前feature的大小是64*64那么与原图的缩放就是8  因为64*8=512
    那么 (y.astype(dtype) + offset) * step就将坐标缩放到0-512的范围了 
    然后(x.astype(dtype) + offset) * step / img_shape[0] 就得到百分比坐标了
    '''
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]
    # 扩展一个维度 exg:64×64*1
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    '''
    下面是将不同形状的长宽（h w）生成出来，其中有不同size和ratios两种变化
    所有的框都是以上面生成的x y作为中心点
    例如当前层的有两种size  (20.48, 51.2)  两种长宽比ratios = [2, 0.5] 那么不同的长宽比就是四种
    注意这里不同ratios不是对应每一个size 而是只对应第一个size 第二个size只有一种长宽比
    '''
    '''初始化'''
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)

    '''第一种框是以size[0]为长宽的正方形框, / img_shape[0]是为了转化为在原图上的百分比'''
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]

    '''第二种框是sqrt(sizes[0] * sizes[1])的正方形框'''
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1

    '''后面第2、3、4种长宽比的框是根据ratios来进行的，
    之所以一个乘sqrt(r) 一个除sqrt(r) 是因为这样可以做到缩进长宽后面积不变'''
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)

    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 512.
# =========================================================================== #
def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_512_vgg'):
    """SSD net definition.
    """

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_512_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        # print("====net.shape = {}".format(net))
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.txt.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net
        if IS_FPN:
            stride_b8_to_b12=1
        else:
            stride_b8_to_b12=2
        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=stride_b8_to_b12, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=stride_b8_to_b12, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=stride_b8_to_b12, scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        end_point = 'concat'
        with tf.variable_scope(end_point):
            high_feature_list = []
            for i in range(9,12):
                high_feature_list.append(end_points['block'+str(i)])
            high_feature = tf.concat(high_feature_list, 1)
            end_points[end_point] = high_feature

        end_point = 'block4_concat'
        with tf.variable_scope(end_point):
            net_block4 = slim.conv2d(end_points['concat'], 128, [1, 1], scope='conv1x1')
            # print("net_block4 ========{}".format(net_block4))
            net_block4 = tf.layers.conv2d_transpose(net_block4,512,kernel_size=3,strides=(4, 4), data_format='channels_first')
            # print("net_block4 ========{}".format(net_block4))
            end_points['block4'] += net_block4

        end_point = 'block7_concat'
        with tf.variable_scope(end_point):
            net_block7 = slim.conv2d(end_points['concat'], 128, [1, 1], scope='conv1x1')
            # print("net_block4 ========{}".format(net_block4))
            net_block7 = tf.layers.conv2d_transpose(net_block7,1024,kernel_size=1,strides=(2, 2), data_format='channels_first')
            # print("net_block4 ========{}".format(net_block4))
            end_points['block7'] += net_block7

        end_point = 'block8_concat'
        with tf.variable_scope(end_point):
            net_block8 = slim.conv2d(end_points['concat'], 512, [1, 1], scope='conv1x1')
            end_points['block8'] += net_block8
        # with tf.variable_scope(end_point):

        # print("=debug===========high_feature = {}".format(high_feature))

        # for i in range(1,12):
           # print("=debug===== "+str(i)+"  = {}".format(end_points['block'+str(i)]))
        # Prediction and localisations layers.
        predictions_pest = []
        logits_pest = []
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                # print("=debug===== "+str(i)+"  = {}".format(end_points[layer]))
                # 这里获取每一个anchor的预测值
                pest, p, l = ssd_vgg_300.ssd_multibox_layer(end_points[layer],
                                                      num_classes,
                                                      anchor_sizes[i],
                                                      anchor_ratios[i],
                                                      normalizations[i])
            predictions_pest.append(prediction_fn(pest))
            logits_pest.append(pest)

            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions_pest, logits_pest, predictions, localisations, logits, end_points
ssd_net.default_image_size = 512


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
'''
ssd_net.losses(logits_pest, b_gclasses_pest, logits, localisations,
                           b_gclasses, b_glocalisations, b_gscores,b_gscores_pest,
                           match_threshold=FLAGS.match_threshold,
                           negative_ratio=FLAGS.negative_ratio,
                           alpha=FLAGS.loss_alpha,
                           label_smoothing=FLAGS.label_smoothing)
'''
def ssd_losses(logits_pest, gclasses_pest, logits, localisations,gclasses, glocalisations, gscores, b_gscores_pest,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    """Loss functions for training the SSD 300 VGG network.

    This function defines the different loss components of the SSD, and
    adds them to the TF loss collection.

    Arguments:
      logits: (list of) predictions logits Tensors;
      localisations: (list of) localisations Tensors;
      gclasses: (list of) groundtruth labels Tensors;
      glocalisations: (list of) groundtruth localisations Tensors;
      gscores: (list of) groundtruth score Tensors;
    """
    # print "=================ssd_loses start======================="
    # print("====match_threshold={}".format(match_threshold))

    with tf.name_scope(scope, 'ssd_losses'):
        l_cross_pos_pest = []
        l_cross_pos = []
        l_cross_neg = []
        l_cross_neg_pest = []
        l_loc = []
        # print "logits = {}".format(logits)
        # i 代表了第几层的输出
        for i in range(len(logits)):
            dtype = logits[i].dtype
            with tf.name_scope('block_%i' % i):

                # Determine weights Tensor.
                label_mask = gscores[i] > match_threshold
                flabel_mask = tf.cast(label_mask, dtype)


                pmask_pest = b_gscores_pest[i] > match_threshold
                fpmask_pest = tf.cast(pmask_pest, dtype)
                n_positives = tf.reduce_sum(fpmask_pest)# 不写维度是多少就是直接把所有的数值相加

                # Negative mask.
                no_classes = tf.cast(pmask_pest, tf.int32)
                predictions = slim.softmax(logits[i])
                nmask = tf.logical_and(tf.logical_not(pmask_pest),# 选出负样本
                                       b_gscores_pest[i] > -0.5)

                fnmask = tf.cast(nmask, dtype)
                nvalues = tf.where(nmask,
                                   predictions[:, :, :, :, 0],
                                   1. - fnmask) #选出来负样本位置的背景预测值 然后其他的地方设置成0
                nvalues_flat = tf.reshape(nvalues, [-1])

                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)

                val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                minval = val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, -nvalues > minval)
                fnmask = tf.cast(nmask, dtype)


                # def add_summary(name,tensor):
                #     op = tf.summary.tensor_summary(name, tensor, collections=[])
                #     op = tf.Print(op, [tensor], name,summarize=500)
                #     tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
                #
                # if i==1:
                #     with tf.name_scope("cross_tropy_debug_summary"):
                #         # summary add for debug
                #         # add_summary("cross_entropy_logits_"+str(i), logits[i])
                #         # add_summary("cross_entropy_logits_pest_" + str(i), logits_pest[i])
                #
                #         add_summary("gscores_" + str(i), gscores[i])
                #         add_summary("gclasses_pest" + str(i), gclasses_pest[i])
                #         add_summary("gclasses_" + str(i), gclasses[i])
                #         add_summary("no_classes_" + str(i), no_classes)
                #         add_summary("flabel_mask_" + str(i), flabel_mask)
                #         add_summary("fnmask_" + str(i), fnmask)
                #         add_summary("fpmask_" + str(i), fpmask_pest)



                with tf.name_scope('cross_entropy_pos'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=gclasses[i]) # gclasses是包含类别的
                    loss = tf.losses.compute_weighted_loss(loss, flabel_mask)
                    l_cross_pos.append(loss)



                with tf.name_scope('cross_entropy_neg'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],
                                                                          labels=no_classes) # no_classes只包含是否为前景
                    loss = tf.losses.compute_weighted_loss(loss, fnmask)
                    l_cross_neg.append(loss)



                with tf.name_scope('cross_entropy_pos_pest'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pest[i],
                                                                          labels=gclasses_pest[i]) # gclasses是包含类别的
                    loss = tf.losses.compute_weighted_loss(loss, fpmask_pest)
                    l_cross_pos_pest.append(loss)
                    # summary add for debug


                with tf.name_scope('cross_entropy_neg_pest'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_pest[i],
                                                                          labels=no_classes) # no_classes只包含是否为前景
                    loss = tf.losses.compute_weighted_loss(loss, fnmask)
                    l_cross_neg_pest.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask_pest, axis=-1)
                    loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                    loss = tf.losses.compute_weighted_loss(loss, weights)
                    l_loc.append(loss)

        # Additional total losses...
        with tf.name_scope('total'):
            total_cross_pos_pest = tf.add_n(l_cross_pos_pest, 'cross_entropy_pos_pest')
            total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
            total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
            total_cross_neg_pest = tf.add_n(l_cross_neg_pest, 'cross_entropy_neg_pest')
            total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
            total_loc = tf.add_n(l_loc, 'localization')

            # Add to EXTRA LOSSES TF.collection total_cross_pos_pest
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg_pest)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos_pest)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', total_cross)
            tf.add_to_collection('EXTRA_LOSSES', total_loc)
        # print "=================ssd_loses end======================="