# -*- coding: utf-8 -*-
import os
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from nets import ssd_vgg_512, np_methods_pest
from preprocessing import ssd_vgg_preprocessing
from pylab import *
import pylab
import imageio
import skimage.io
import cv2
from cStringIO import StringIO
import PIL
import tf_utils
from notebooks import visualization

slim = tf.contrib.slim


class SsdDetector:
    def __init__(self,config):
        # Default parameter
        self.VOC_LABELS = {
            'none': (0, 'Background'),
            'sz': (1, 'sz'),
            'ls': (2, 'ls'),
            'tc': (3, 'tc'),
            'rd': (4, 'rd'),
            'os': (5, 'os'),
            'cp': (6, 'cp'),
        }
        self.name_list = ['sz', 'ls', 'tc', 'rd', 'os', 'cp']
        self.net_shape = (512, 512)
        self.data_format = 'NCHW'
        self.ckpt_filename = config.detector_model_path

        # Create Session
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=config)

        # Create process
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = \
            ssd_vgg_preprocessing.preprocess_for_eval(
                self.img_input,
                None,
                None,
                self.net_shape,
                self.data_format,
                resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)

        # Define the SSD model.
        reuse = True if 'ssd_net' in locals() else None
        self.ssd_net = ssd_vgg_512.SSDNet()
        with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
            self.predictions_pest, self.logits_pest, self.predictions, self.localisations, _, _ = \
                self.ssd_net.net(
                    self.image_4d,
                    is_training=False,
                    reuse=reuse)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.ckpt_filename)

        # SSD default anchor boxes.
        self.ssd_anchors = self.ssd_net.anchors(self.net_shape)

    def process_image(self, img, select_threshold, nms_threshold):
        # Run SSD network.
        rimg, rpredictions_pest, rpredictions, rlocalisations, rbbox_img = \
            self.sess.run([self.image_4d, self.predictions_pest, self.predictions, self.localisations, self.bbox_img],
                          feed_dict={self.img_input: img})

        rclasses, rclasses_category, rscores_category, rscores, rbboxes = np_methods_pest.ssd_bboxes_select(
            rpredictions_pest, rpredictions, rlocalisations, self.ssd_anchors,
            select_threshold=select_threshold, img_shape=self.net_shape, num_classes=6, decode=True)

        rbboxes = np_methods_pest.bboxes_clip(rbbox_img, rbboxes) #
        rclasses, rscores, rbboxes, rclasses_category, rscores_category = \
            np_methods_pest.bboxes_sort(rclasses, rscores, rbboxes, rclasses_category, rscores_category, top_k=400)
        rclasses, rscores, rbboxes, rclasses_category, rscores_category = \
            np_methods_pest.bboxes_nms(rclasses, rscores, rbboxes, rclasses_category, rscores_category, nms_threshold=nms_threshold)

        rbboxes = np_methods_pest.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes, rclasses_category, rscores_category


    def get_xml_ground_truth(self,xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        name = None
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            bboxes.append((int(bbox.find('ymin').text),
                           int(bbox.find('xmin').text),
                           int(bbox.find('ymax').text),
                           int(bbox.find('xmax').text)
                           ))
        return name, bboxes

    def plt_temp_cv2(self, img, x1_axis, y1_axis, x2_axis, y2_axis):
        for i,_ in enumerate(x1_axis):
            # print y1_axis[i], x1_axis[i], y2_axis[i], x2_axis[i]
            cv2.rectangle(img, (x1_axis[i], y1_axis[i]), (x2_axis[i], y2_axis[i]), (0, 255, 0), 1)
        return img

    def plt_bboxes_cv2(self, img, classes, scores, bboxes, rclasses_category, rscores_category, GT_name=None, GT_bboxes=None,
                       figsize=(10, 10), linewidth=1.5):

        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(classes.shape[0]):
            cls_id = int(rclasses_category[i])
            if cls_id >= 0:
                score = scores[i]
                score_category = rscores_category[i][0]

                if cls_id not in colors:
                    colors[cls_id] = [255, 0, 0]
                ymin = int(bboxes[i, 0] * height)
                xmin = int(bboxes[i, 1] * width)
                ymax = int(bboxes[i, 2] * height)
                xmax = int(bboxes[i, 3] * width)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                class_name = self.name_list[cls_id - 1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, '{:s} | {:.3f} |{:.3f}'.format(class_name, score, score_category),
                            (xmin, ymin - 2), font, 0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        # GT
        if GT_name != None and GT_bboxes != None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, '{:s} '.format(str(self.VOC_LABELS[GT_name][0])),
                        (10, 10), font, 0.3, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            for GT_bbox in GT_bboxes:
                ymin = int(GT_bbox[0])
                xmin = int(GT_bbox[1])
                ymax = int(GT_bbox[2])
                xmax = int(GT_bbox[3])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [255, 0, 0], 1)
        return img


    def predict(self,img,select_threshold=0.9, nms_threshold=0.3,xml_path=None):
        class_statistic_dict = dict()
        rclasses, rscores, rbboxes,rclasses_category, rscores_category = \
            self.process_image(img, select_threshold, nms_threshold)
        if xml_path:
            GT_name, GT_bboxes = self.get_xml_ground_truth(xml_path)
        else:
            GT_name = None
            GT_bboxes = None

        img = self.plt_bboxes_cv2(img, rclasses, rscores, rbboxes,rclasses_category, rscores_category, GT_name, GT_bboxes)
        for i in range(rclasses_category.shape[0]):
            if rclasses_category[i][0] in class_statistic_dict:
                class_statistic_dict[rclasses_category[i][0]] +=1
            else:
                class_statistic_dict[rclasses_category[i][0]] = 1
        return img,class_statistic_dict

    def predict_full_pic(self,img,select_threshold=0.9, nms_threshold=0.3,kaobian_thresh = 5,xml_path=None):
        # temp statistic
        # x1_axis = []
        # y1_axis = []
        # x2_axis = []
        # y2_axis = []

        class_statistic_dict = dict()
        img_h, img_w, _ = img.shape
        # print "img_h = {}".format(img_h)
        # print "img_w = {}".format(img_w)
        num_h = int(img_h / 452)
        num_w = int(img_w / 452)
        rclasses_big = []
        rscores_big = []
        rclasses_category_big = []
        rscores_category_big = []
        rbboxes_big = []
        for i in range(num_w+1):# x坐标
            for j in range(num_h+1):# y坐标
                if i==0:
                    x_jiange = 0
                else:
                    x_jiange = -60
                if j==0:
                    y_jiange = 0
                else:
                    y_jiange = -60

                x1, y1, x2, y2 = i * (512+x_jiange), j * (512+y_jiange), i * (512+x_jiange) + 512, j * (512+y_jiange) + 512
                if x2> img_w:
                    x1 = img_w-512
                    x2 = img_w
                if y2 > img_h:
                    y1 = img_h-512
                    y2 = img_h

                sub_img = img[y1:y2, x1:x2, :]
                # print i,j
                # print x1,y1,x2,y2
                # x1_axis.append(x1)
                # y1_axis.append(y1)
                # x2_axis.append(x2)
                # y2_axis.append(y2)
                # print
                rclasses, rscores, rbboxes, rclasses_category, rscores_category = self.process_image(sub_img,select_threshold, nms_threshold)
                # visualization.plt_bboxes(sub_img, rclasses_category, rscores_category, rbboxes)
                for ii in range(rbboxes.shape[0]):
                    ymin = int(rbboxes[ii, 0] * 512) + y1
                    xmin = int(rbboxes[ii, 1] * 512) + x1
                    ymax = int(rbboxes[ii, 2] * 512) + y1
                    xmax = int(rbboxes[ii, 3] * 512) + x1

                    if (ymin - y1 < kaobian_thresh and j != 0) or (xmin - x1 < kaobian_thresh and i != 0) \
                            or (x2 - xmax < kaobian_thresh and i != num_h - 1) or (
                                y2 - ymax < kaobian_thresh and j != num_w - 1):
                        pass
                    else:

                        rbboxes[ii, 0] = (ymin + 0.0) / (img_h + 0.0)
                        rbboxes[ii, 1] = (xmin + 0.0) / (img_w + 0.0)
                        rbboxes[ii, 2] = (ymax + 0.0) / (img_h + 0.0)
                        rbboxes[ii, 3] = (xmax + 0.0) / (img_w + 0.0)

                        rbboxes_big.append(rbboxes[ii, :])
                        rclasses_big.append(rclasses[ii])
                        rscores_big.append(rscores[ii])
                        rclasses_category_big.append(rclasses_category[ii])
                        rscores_category_big.append(rscores_category[ii])
                        # print rclasses_category

                        # print "vv"
        rbboxes_big = np.array(rbboxes_big)
        rclasses_big = np.array(rclasses_big)
        rscores_big = np.array(rscores_big)
        rclasses_category_big = np.array(rclasses_category_big)
        rscores_category_big = np.array(rscores_category_big)
        img_detected = self.plt_bboxes_cv2(img, rclasses_big, rscores_big, rbboxes_big, rclasses_category_big,
                                      rscores_category_big)
        # img_detected = self.plt_temp_cv2(img_detected,x1_axis,y1_axis,x2_axis,y2_axis)
        for i in range(rclasses_category_big.shape[0]):
            if rclasses_category_big[i][0] in class_statistic_dict:
                class_statistic_dict[rclasses_category_big[i][0]] +=1
            else:
                class_statistic_dict[rclasses_category_big[i][0]] = 1
        return img_detected,class_statistic_dict

if __name__ == '__main__':
    detector = SsdDetector("../../model/detector/model.ckpt-928108")
    img = mpimg.imread('../../client_example/full_pic.jpg')
    img,category = detector.predict_full_pic(img)
    print category
    # print img
    # cv2.imshow("ss",img)
    # cv2.waitKey(0)