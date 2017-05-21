# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:46:47 2017

@author: yuanqi

KNN, Tensorflow Version
"""

import numpy as np
import tensorflow as tf
import pandas as pd

def load_data_set(filename = '../input/train.csv', train_propotion = 1.0):
    """
    filename: 训练集路径
    train_propotion: 验证集所占比例    
    返回训练集和验证集
    """
    train_set = pd.read_csv('../input/train.csv')
    train_num = int(train_set.shape[0] * train_propotion)
    
    label = np.array(train_set.iloc[:, 0])
    pixel_mat = np.array(train_set.iloc[:, 1:])
    
    train_label = label[:train_num]
    train_pixel_mat = pixel_mat[:train_num]
    ret_label = []
    for l in train_label:
        tmp = [0.] * 10
        tmp[int(l)] = 1.
        ret_label.append(tmp)
    #valid_label = label[train_num:]
    #valid_pixel_mat = pixel_mat[train_num:]

    return np.array(train_pixel_mat), np.array(ret_label)#, valid_pixel_mat, valid_label
    
def load_test_set(filename = '../input/test.csv'):
    """
    返回测试集
    """
    test_set = pd.read_csv(filename)        
    pixel_mat = np.array(test_set.iloc[:,:])
    return pixel_mat
    
train_pixels, train_labels = load_data_set()
test_pixels = load_test_set()

W = tf.placeholder("float", [None, 784])
b = tf.placeholder("float", [784])

dist = tf.reduce_sum(tf.abs(tf.add(W, tf.negative(b))), reduction_indices=1)
pred = tf.arg_min(dist, 0)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    filename = "new.csv"
    f = open(filename, "wa")
    f.write("ImageId,Label\n")    
    for i in range(len(test_pixels)):        
        index = sess.run(pred, feed_dict={W: train_pixels, b: train_labels[i, :]})        
        label = np.argmax(train_labels[index])        
        f.write("{0},{1}\n".format(i + 1, label))        
    f.close()
    print("Done!")