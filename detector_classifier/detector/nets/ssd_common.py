# -*- coding: utf-8 -*-
# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Shared function between different SSD implementations.
"""
import numpy as np
import tensorflow as tf
import tf_extended as tfe


# =========================================================================== #
# TensorFlow implementation of boxes SSD encoding / decoding.
# =========================================================================== #
def tf_ssd_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using SSD anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    # print "=============tf_ssd_bboxes_encode_layer===begin================="
    # print "labels = {}".format(labels)
    # print "bboxes = {}".format(bboxes)
    # print "anchors_layer = {}".format(anchors_layer)
    # print "num_classes = {}".format(num_classes)
    # print "prior_scaling = {}".format(prior_scaling)

    # yref...是anchor的中心化的坐标 就是yref xref是anchor的中心点坐标
    # href wref anchor中心往外的高度和宽度
    yref, xref, href, wref = anchors_layer
    # print "yref.shape= {} xref.shape= {} href.shape= {} wref.shape= {}  ".\
    # format(yref.shape, xref.shape, href.shape, wref.shape)
    # ymin xmin ymax xmax是左上角和右上角的坐标
    # yref.shape= (32, 32, 1) xref.shape= (32, 32, 1) href.shape= (6,) wref.shape= (6,)
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    # 每个anchor的面积
    vol_anchors = (xmax - xmin) * (ymax - ymin)
    # print "ymin.shape= {} xmin.shape= {} ymax.shape= {} xmax.shape= {}  ". \
    # format(ymin.shape, xmin.shape, ymax.shape, xmax.shape)
    # print "vol_anchors= {}".format(vol_anchors)
    # Initialize tensors...
    # exg: 64×64×4  当前layer有64*64个框 然后有4种不同大小的框
    shape = (yref.shape[0], yref.shape[1], href.size)
    # print "shape= {}".format(shape)
    feat_labels_pest = tf.zeros(shape, dtype=tf.int64)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)
    feat_scores_pest = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)
    one_constant = tf.constant(value=1,dtype=tf.int64)


    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
           所有的操作都是tensor的广播模式
           当前的GT框与所有anchor  IOU
        """
        # ymin xmin ymax xmax是所有的anchar的坐标
        # 下面4行是算出 当前的GT框的坐标和所有anchor相交区域的坐标
        # int_ymin的形状是B*64*64*4
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        # 每个anchor与GT的相交区域的长宽  若相交区域的长宽是负数那么就是不想交 取0
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w # 每个anchor与GT的相交区域的面积 anchor交GT(inter_vol)
        # union_vol(anchor并GT) = 每个anchor的面积 + bbox的面积 - anchor与bbox相交的面积
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i, feat_labels_pest, feat_labels, feat_scores_pest,feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels_pest, feat_labels, feat_scores_pest,feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[i]
        bbox = bboxes[i]
        # 当前bbox与所有anchor的IOU值 形状与anchor的形状一致 B×64*64*4
        jaccard = jaccard_with_anchors(bbox)

        # 不包含类别7的
        mask = tf.greater(jaccard, feat_scores)
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes) #通过这个过滤掉类别7的
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        feat_scores = tf.where(mask, jaccard, feat_scores)
        feat_labels = imask * label + (1 - imask) * feat_labels

        # 包含类别7的
        mask_pest = tf.greater(jaccard, feat_scores_pest)
        mask_pest = tf.logical_and(mask_pest, feat_scores_pest > -0.5)
        imask_pest = tf.cast(mask_pest, tf.int64)
        fmask_pest = tf.cast(mask_pest, dtype)

        feat_scores_pest = tf.where(mask_pest, jaccard, feat_scores_pest)
        feat_labels_pest  = imask_pest * one_constant + (1 - imask_pest) * feat_labels_pest

        feat_ymin = fmask_pest * bbox[0] + (1 - fmask_pest) * feat_ymin
        feat_xmin = fmask_pest * bbox[1] + (1 - fmask_pest) * feat_xmin
        feat_ymax = fmask_pest * bbox[2] + (1 - fmask_pest) * feat_ymax
        feat_xmax = fmask_pest * bbox[3] + (1 - fmask_pest) * feat_xmax


        return [i+1, feat_labels_pest, feat_labels, feat_scores_pest,feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]

    i = 0
    [i,feat_labels_pest, feat_labels, feat_scores_pest,feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_labels_pest, feat_labels, feat_scores_pest,feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    # 这是将左上角右上角的坐标方式 转化为中心坐标和长宽坐标的方式
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    # 这里有疑问： 对于feat_cy的地方那么减完就是负数了
    # feat_cy是与GT相交的GT的坐标 yref href 是anchor的坐标 下面四行是做的偏移量转化
    # prior_scaling[0]论文中没有这个 估计是一个缩放
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    # print "=============tf_ssd_bboxes_encode_layer===end================="
    return feat_labels_pest, feat_labels, feat_localizations, feat_scores,feat_scores_pest


def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int64) containing groundtruth labels;
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_labels_pest = []
        target_localizations = []
        target_scores = []
        target_scores_pest = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                #feat_labels_pest, feat_labels, feat_localizations, feat_scores,feat_scores_pest
                t_labels_pest, t_labels, t_loc, t_scores,t_scores_pest = tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, dtype)

                target_labels_pest.append(t_labels_pest)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
                target_scores_pest.append(t_scores_pest)
        return target_labels_pest, target_labels, target_localizations, target_scores,target_scores_pest


def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes


# =========================================================================== #
# SSD boxes selection.
# =========================================================================== #
def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=7,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # shape = batch_size × (64*64*4) × number_classes
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        # print "debug+++++++++++++++predictions_layer = {} ".format(predictions_layer)
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))
        # print "localizations_layer = {} ".format(localizations_layer)
        d_scores = {}
        d_bboxes = {}
        # for c in range(0, num_classes):
        for c in range(0, 2):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def tf_ssd_bboxes_select(predictions_net, localizations_net,
                         select_threshold=None,
                         num_classes=21,
                         ignore_class=0,
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        select_threshold,
                                                        num_classes,
                                                        ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            # print "c = {}".format(c)
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            # print "ls = {}".format(ls)
            # print "lb = {}".format(lb)
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
            # print "d_bboxes[c] = {}".format(d_bboxes[c])
            # print "d_scores[c] = {}".format(d_scores[c])
            # print ""
        return d_scores, d_bboxes


def tf_ssd_bboxes_select_layer_all_classes(predictions_layer, localizations_layer,
                                           select_threshold=None):
    """Extract classes, scores and bounding boxes from features in one layer.
     Batch-compatible: inputs are supposed to have batch-type shapes.

     Args:
       predictions_layer: A SSD prediction layer;
       localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
     Return:
      classes, scores, bboxes: Input Tensors.
     """
    # Reshape features: Batches x N x N_labels | 4
    p_shape = tfe.get_shape(predictions_layer)
    predictions_layer = tf.reshape(predictions_layer,
                                   tf.stack([p_shape[0], -1, p_shape[-1]]))
    l_shape = tfe.get_shape(localizations_layer)
    localizations_layer = tf.reshape(localizations_layer,
                                     tf.stack([l_shape[0], -1, l_shape[-1]]))
    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = tf.argmax(predictions_layer, axis=2)
        scores = tf.reduce_max(predictions_layer, axis=2)
        scores = scores * tf.cast(classes > 0, scores.dtype)
    else:
        sub_predictions = predictions_layer[:, :, 1:]
        classes = tf.argmax(sub_predictions, axis=2) + 1
        scores = tf.reduce_max(sub_predictions, axis=2)
        # Only keep predictions higher than threshold.
        mask = tf.greater(scores, select_threshold)
        classes = classes * tf.cast(mask, classes.dtype)
        scores = scores * tf.cast(mask, scores.dtype)
    # Assume localization layer already decoded.
    bboxes = localizations_layer
    return classes, scores, bboxes


def tf_ssd_bboxes_select_all_classes(predictions_net, localizations_net,
                                     select_threshold=None,
                                     scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
    Return:
      classes, scores, bboxes: Tensors.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_classes = []
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            classes, scores, bboxes = \
                tf_ssd_bboxes_select_layer_all_classes(predictions_net[i],
                                                       localizations_net[i],
                                                       select_threshold)
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)

        classes = tf.concat(l_classes, axis=1)
        scores = tf.concat(l_scores, axis=1)
        bboxes = tf.concat(l_bboxes, axis=1)
        return classes, scores, bboxes

