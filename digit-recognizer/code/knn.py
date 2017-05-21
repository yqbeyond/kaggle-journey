# -*- coding: utf-8 -*-
"""
Created on Fri May 19 21:28:59 2017

@author: yuanqi

Stupid KNN Version.
"""

import numpy as np
import pandas as pd
from PIL import Image
import time

def get_eig(data_mat):
    """
    返回特征值和特征向量
    """
    mean_vals = np.mean(data_mat, axis = 0)
    mean_removed = data_mat - mean_vals
    cov_mat = np.cov(mean_removed, rowvar = 0)
    eigvals, eigvects = np.linalg.eig(np.mat(cov_mat))    
    return eigvals, eigvects
    
def pca(data_mat, acc = 0.99):
    """
    降维：返回降维后的矩阵和特征向量（事实证明降维效果太差了, 0.99的精确度降到了710维）
    """
    mean_vals = np.mean(data_mat, axis = 0)
    mean_removed = data_mat - mean_vals
    eigvals, eigvects = get_eig(data_mat)
    eigval_ind = np.argsort(eigvals)
    var_vals = np.var(data_mat, axis = 0)
    var_vals = np.array(var_vals)[0]
    sum_var_vals = sum(var_vals)
    feature_num = None
    tmp_sum = 0.
    for i in range(data_mat.shape[1]):
        tmp_sum += var_vals[i]
        if tmp_sum / sum_var_vals > acc:
            feature_num = i
            break

    eigval_ind = eigval_ind[:-(feature_num + 1): -1]    
    red_eig_vects = eigvects[:, eigval_ind]                            
    low_data_mat = mean_removed * red_eig_vects
    return low_data_mat,red_eig_vects
    
def low_input_vec(low_data_mat, eig_vects, input_vec):
    """
    返回降维后的输入变量
    """
    mean = np.mean(input_vec)
    mean_removed = input_vec - mean
    ret_vec = mean_removed * eig_vects
    return ret_vec

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
    
    valid_label = label[train_num:]
    valid_pixel_mat = pixel_mat[train_num:]

    return train_pixel_mat, train_label, valid_pixel_mat, valid_label
    
def load_test_set(filename = '../input/test.csv'):
    """
    返回测试集
    """
    test_set = pd.read_csv(filename)        
    pixel_mat = np.array(test_set.iloc[:,:])
    return pixel_mat
    

def knn(pixel_mat, label, input_vec, k = 10):
    """
    返回分类结果
    """
    dist = []
    for pixel_row in pixel_mat:
        euclidean_dist = np.sqrt(sum(np.square(input_vec - pixel_row))) # 计算欧式距离
        dist.append(euclidean_dist)
        
    label_dist_pair = list(zip(label, dist))
    label_dist_pair.sort(key = lambda i: i[1])
    
    top_k = np.array(label_dist_pair[:k])
    label_set = set(top_k[:,0])
    label_count = dict(zip(label_set, [0] * len(label_set)))
    for item in top_k:
        label_count[item[0]] += 1
    label = max(label_count.items(), key = lambda item: item[1])[0]
    return int(label)#, label_count

def generate_pic(input_vec, filename):
    """
    根据输入向量产生图片
    """
    data_mat = []
    for i in range(0, input_vec.shape[0], 28):
        data_mat.append(input_vec[i:i+28])        
    img = Image.fromarray(np.array(np.uint8(data_mat)))
    img.save(filename)

def validate(train_pixel_mat, train_label, valid_pixel_mat, valid_label):    
    right_predict = 0.    
    for i, valid_item in enumerate(valid_pixel_mat):
        label = knn(train_pixel_mat, train_label, valid_item)
        print (label, valid_label[i])
        if label == valid_label[i]:
            right_predict += 1.
    acc = right_predict / valid_pixel_mat.shape[0]    
    return acc

def test(test_pixel_mat, train_pixel_mat, train_label, filename="result.csv"):
    """
    生成训练结果保存为csv文件
    """
    f = open(filename, "w")
    f.write("ImageId,Label\n")
    for i, item in enumerate(test_pixel_mat):
        label = knn(train_pixel_mat, train_label, item)
        #print("{0},{1}".format(i + 1, label))
        f.write("{0},{1}\n".format(i + 1, label))

if __name__ == "__main__":
    start = time.time()
    train_pixel_mat, train_label, valid_pixel_mat, valid_label = load_data_set()
    #acc = validate(train_pixel_mat, train_label, valid_pixel_mat, valid_label)
    #print (acc)
    test_pixel_mat = load_test_set()
    
    #train_pixel_mat = pca(train_pixel_mat)
    #valid_pixel_mat = pca(valid_pixel_mat)    
    #test_pixel_mat = pca(test_pixel_mat)
    
    #test(test_pixel_mat, train_pixel_mat, train_label, filename="result.csv")
    end = time.time()
    print (start - end)
    