#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a implementation of our LPNet structure:
# X. Fu, B. Liang, Y. Huang, X. Ding and J. Paisley. “Lightweight Pyramid Networks for Image Deraining”, IEEE Transactions on Neural Networks and Learning Systems, 2019.
# author: Xueyang Fu (xyfu@ustc.edu.cn)

import numpy as np
import tensorflow as tf
import scipy.misc
import scipy.io 

num_pyramids = 4 # number of pyramid levels
num_blocks = 5    # number of recursive blocks
num_feature = 16  # number of feature maps
num_channels = 3  # number of input's channels 

DEFAULT_PADDING = 'SAME'
def validate_padding(padding):
        assert padding in ('SAME', 'VALID') 
        
def make_var(name, shape, initializer=None, trainable=True, regularizer=None):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

def l2_regularizer(weight_decay=0.0005, scope=None):
    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(weight_decay,
                                   dtype=tensor.dtype.base_dtype,
                                   name='weight_decay')
            return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
    return regularizer


def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    validate_padding(padding)
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def perceptual_loss(self, real_image, generated_image):
    real_features = self.vgg16_extract_feature(real_image)
    generated_features = self.vgg16_extract_feature(generated_image)
    loss_1 = tf.reduce_mean(tf.abs(
    real_features[0] -
    generated_features[0]))
    loss_2 = tf.reduce_mean(tf.abs(
    real_features[1] -
    generated_features[1]))
    loss = loss_1 + loss_2
    return self.lambda_perceptual * loss


def _tf_fspecial_gauss(size, sigma):
    """ Function to mimic the 'fspecial' gaussian MATLAB functino
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma * 2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map = False, mean_metric = True, size = 11, sigma = 1.5):
    window = _tf_fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides = [1,1,1,1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1,1,1,1], padding = 'VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1,1,1,1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1,1,1,1], padding = 'VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides = [1,1,1,1], padding = 'VALID') -mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def image_to_4d(image):
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    return image

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype = tf.float32))
    return numerator / denominator


def loss_ssim(img1, img2, batchsize, c_dims):
    ssim_value_sum = 0
    for i in range (batchsize):
        for j in range (c_dims):
            img1_tmp = img1[i, :, :, j]
            img1_tmp = image_to_4d(img1_tmp)
            img2_tmp = img2[i, :, :, j]
            img2_tmp = image_to_4d(img2_tmp)
            ssim_value_tmp = tf_ssim(img1_tmp, img2_tmp)
            ssim_value_sum += ssim_value_tmp
    ssim_value_ave = ssim_value_sum / (batchsize * c_dims)
    return log10(1.0 / (ssim_value_ave + 1e-4))






def conv2d(input_, w, b, ks=3, stride = 1, stddev=0.02, name="conv2d_op", reuse = False):
   conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding='SAME', name=name + "_conv_op")
   return tf.add(conv, b, name = name + "_add_op")


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):

    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-4)

    new_variables = tf.get_variable(name=name+"conv", shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def lrelu(x, leak=0.2, name = 'lrelu'):
    
    with tf.variable_scope(name):
        return tf.maximum(x, leak*x, name = name)   

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img




# leaky ReLU
def lrelu(x, leak = 0.2, name = 'lrelu'):   
    with tf.variable_scope(name):
         return tf.maximum(x, leak*x, name = name)   



######## Laplacian and Gaussian Pyramid ########
def lap_split(img,kernel):
    with tf.name_scope('split'):
        low = tf.nn.conv2d(img, kernel, [1,2,2,1], 'SAME')
        low_upsample = tf.nn.conv2d_transpose(low, kernel*4, tf.shape(img), [1,2,2,1])
        high = img - low_upsample
    return low, high

def LaplacianPyramid(img,kernel,n):
    levels = []
    for i in range(n):
        img, high = lap_split(img, kernel)
        levels.append(high)
    levels.append(img)
    return levels[::-1]

def GaussianPyramid(img,kernel,n):
    levels = []
    low = img
    for i in range(n):
        low = tf.nn.conv2d(low, kernel, [1,2,2,1], 'SAME')
        levels.append(low)
    return levels[::-1]
######## Laplacian and Gaussian Pyramid ######## 



# create kernel
def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-4)
    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables





def recursive_block1(rain_img, c_dim = 3, output_channel = 16, recursive_num = 3, index = 1, stride = 1, is_train = True):
    init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG',uniform=False)
    init_biases = tf.constant_initializer(0.0)
    with tf.variable_scope("rec_block_%d" % index):
        w0 = make_var('conv0', shape = [3,3,c_dim,output_channel],initializer = init_weights, trainable = True,regularizer = l2_regularizer(0.0005))
        b0 = make_var('bias0', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w1 = make_var('conv1', shape = [3,3,output_channel,output_channel],initializer = init_weights,trainable = True, regularizer = l2_regularizer(0.0005))
        b1 = make_var('bias1', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        w2 = make_var('conv2', shape = [3,3,output_channel,output_channel],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b2 = make_var('bias2', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w3 = make_var('conv3', shape = [3,3,output_channel,c_dim],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b3 = make_var('bias3', shape = [c_dim], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        conv1_1 = lrelu(conv2d(rain_img, w0, b0))
        conv_temp = conv1_1
        for i in range(recursive_num):
            conv_temp = lrelu(conv2d(conv_temp, w1, b1))
            conv_temp = lrelu(conv2d(conv_temp, w2, b2))
        conv1_2 = conv_temp
        res = lrelu(conv2d(conv1_2, w3, b3))
        return rain_img + res, res
    
def recursive_block2(rain_img, input_, res, c_dim = 3, output_channel = 16, recursive_num = 3, index = 2, stride = 1, is_train = True):
    init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG',uniform=False)
    init_biases = tf.constant_initializer(0.0)
    with tf.variable_scope("rec_block_%d" % index):
        w0 = make_var('conv0', shape = [3,3,c_dim,output_channel],initializer = init_weights, trainable = True,regularizer = l2_regularizer(0.0005))
        b0 = make_var('bias0', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w1 = make_var('conv1', shape = [3,3,output_channel+c_dim,output_channel],initializer = init_weights,trainable = True, regularizer = l2_regularizer(0.0005))
        b1 = make_var('bias1', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        w2 = make_var('conv2', shape = [3,3,output_channel+c_dim,output_channel],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b2 = make_var('bias2', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w3 = make_var('conv3', shape = [3,3,output_channel+c_dim,c_dim],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b3 = make_var('bias3', shape = [c_dim], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        conv1_1 = lrelu(conv2d(input_, w0, b0))
        conv_temp = tf.concat([conv1_1, res], 3, name = 'concat1')
        for i in range(recursive_num):
            conv_temp = lrelu(conv2d(conv_temp, w1, b1))
            maps1 = conv_temp
            conv_temp = tf.concat([conv_temp, res], 3)
            conv_temp = lrelu(conv2d(conv_temp, w2, b2))
            maps2 = conv_temp
            conv_temp = tf.concat([conv_temp, res], 3)
        conv1_2 = conv_temp
        res = lrelu(conv2d(conv1_2, w3, b3))
        return rain_img + res, res
        #return rain_img + res, res, maps1, maps2

def recursive_block3(rain_img, input_, res1, res2, c_dim, output_channel = 16, recursive_num = 3, index = 3, stride = 1, is_train = True):
    init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG',uniform=False)
    init_biases = tf.constant_initializer(0.0)
    with tf.variable_scope("rec_block_%d" % index):
        w0 = make_var('conv0', shape = [3,3,c_dim,output_channel],initializer = init_weights, trainable = True,regularizer = l2_regularizer(0.0005))
        b0 = make_var('bias0', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w1 = make_var('conv1', shape = [3,3,output_channel+2*c_dim,output_channel],initializer = init_weights,trainable = True, regularizer = l2_regularizer(0.0005))
        b1 = make_var('bias1', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        w2 = make_var('conv2', shape = [3,3,output_channel+2*c_dim,output_channel],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b2 = make_var('bias2', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w3 = make_var('conv3', shape = [3,3,output_channel+2*c_dim,c_dim],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b3 = make_var('bias3', shape = [c_dim], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        conv1_1 = lrelu(conv2d(input_, w0, b0))
        conv_temp = tf.concat([conv1_1, res1, res2], 3)
        for i in range(recursive_num):
            conv_temp = lrelu(conv2d(conv_temp, w1, b1))
            conv_temp = tf.concat([conv_temp, res1, res2], 3)
            conv_temp = lrelu(conv2d(conv_temp, w2, b2))
            conv_temp = tf.concat([conv_temp, res1, res2], 3)

        conv1_2 = conv_temp
        res = lrelu(conv2d(conv1_2, w3, b3))
        return rain_img + res, res
    
def recursive_block4(rain_img, input_, res1, res2, res3, c_dim, output_channel = 16, recursive_num = 3, index = 4, stride = 1, is_train = True):
    init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG',uniform=False)
    init_biases = tf.constant_initializer(0.0)
    with tf.variable_scope("rec_block_%d" % index):
        w0 = make_var('conv0', shape = [3,3,c_dim,output_channel],initializer = init_weights, trainable = True,regularizer = l2_regularizer(0.0005))
        b0 = make_var('bias0', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w1 = make_var('conv1', shape = [3,3,output_channel+3*c_dim,output_channel],initializer = init_weights,trainable = True, regularizer = l2_regularizer(0.0005))
        b1 = make_var('bias1', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        w2 = make_var('conv2', shape = [3,3,output_channel+3*c_dim,output_channel],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b2 = make_var('bias2', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w3 = make_var('conv3', shape = [3,3,output_channel+3*c_dim,c_dim],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b3 = make_var('bias3', shape = [c_dim], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        conv1_1 = lrelu(conv2d(input_, w0, b0))
        conv_temp = tf.concat([conv1_1, res1, res2, res3], 3)
        for i in range(recursive_num):
            conv_temp = lrelu(conv2d(conv_temp, w1, b1))
            conv_temp = tf.concat([conv_temp, res1, res2, res3], 3)
            conv_temp = lrelu(conv2d(conv_temp, w2, b2))
            conv_temp = tf.concat([conv_temp, res1, res2, res3], 3)

        conv1_2 = conv_temp
        res = lrelu(conv2d(conv1_2, w3, b3))
        return rain_img + res, res

def recursive_block5(rain_img, input_, res1, res2, res3, res4, c_dim, output_channel = 16, recursive_num = 3, index = 5, stride = 1, is_train = True):
    init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG',uniform=False)
    init_biases = tf.constant_initializer(0.0)
    with tf.variable_scope("rec_block_%d" % index):
        w0 = make_var('conv0', shape = [3,3,c_dim,output_channel],initializer = init_weights, trainable = True,regularizer = l2_regularizer(0.0005))
        b0 = make_var('bias0', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w1 = make_var('conv1', shape = [3,3,output_channel+4*c_dim,output_channel],initializer = init_weights,trainable = True, regularizer = l2_regularizer(0.0005))
        b1 = make_var('bias1', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        w2 = make_var('conv2', shape = [3,3,output_channel+4*c_dim,output_channel],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b2 = make_var('bias2', shape = [output_channel], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        w3 = make_var('conv3', shape = [3,3,output_channel+4*c_dim,c_dim],initializer = init_weights,trainable = True,regularizer = l2_regularizer(0.0005))
        b3 = make_var('bias3', shape = [c_dim], initializer = init_biases, trainable = True, regularizer = l2_regularizer(0.0005))
        
        conv1_1 = lrelu(conv2d(input_, w0, b0))
        conv_temp = tf.concat([conv1_1, res1, res2, res3, res4], 3)
        for i in range(recursive_num):
            conv_temp = lrelu(conv2d(conv_temp, w1, b1))
            conv_temp = tf.concat([conv_temp, res1, res2, res3, res4], 3)
            conv_temp = lrelu(conv2d(conv_temp, w2, b2))
            conv_temp = tf.concat([conv_temp, res1, res2, res3, res4], 3)

        conv1_2 = conv_temp
        res = lrelu(conv2d(conv1_2, w3, b3))
        return rain_img + res, res
    
def subnet(rain_img, c_dim, out_channel, num = 5, is_train = True, reuse = False):
    with tf.variable_scope('recursive_net2'):
        if reuse:
            tf.get_variable_scope().reuse_variables()
#        result = gauss_layer(input_, c_dim, is_train = is_train)
        output1, res1 = recursive_block1(rain_img, c_dim, out_channel, recursive_num = num)
        output2, res2 = recursive_block2(rain_img, output1, res1, c_dim, out_channel, recursive_num = num)
        output3, res3 = recursive_block3(rain_img, output2, res1, res2, c_dim, out_channel, recursive_num = num)
        output4, res4 = recursive_block4(rain_img, output3, res1, res2, res3, c_dim, out_channel, recursive_num = num)
        output5, res5 = recursive_block5(rain_img, output4, res1, res2, res3, res4, c_dim, out_channel, recursive_num = num)
       # output = conv2d(tf.concat([output1, output2, output3, output4, output5], 2, name = 'concat'),3,3,3,1,1,name='output')
#        output=(output1+output2+output3+output4+output5)/5
        final = tf.concat([output5, output4, output3, output2,output1], 3)
    return output4

    # LPNet structure










def inference(images):
    with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
        k = np.float32([.0625, .25, .375, .25, .0625])  # Gaussian kernel for image pyramid
        k = np.outer(k, k)
        kernel = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
        pyramid = LaplacianPyramid(images, kernel, (num_pyramids - 1))  # rainy Laplacian pyramid

        # subnet 1
        with tf.variable_scope('subnet1'):
            out1 = subnet(pyramid[0], num_channels,int((num_feature) / 16))
            
            out1 = tf.nn.relu(out1)
           # out1=lrelu(out1)
            out1_t = tf.nn.conv2d_transpose(out1, kernel * 4, tf.shape(pyramid[1]), [1, 2, 2, 1])

        # subnet 2
        with tf.variable_scope('subnet2'):
            out2 = subnet(pyramid[1], num_channels,int((num_feature) / 8))
            out2 = tf.add(out2, out1_t)
           # out2 = lrelu(out2)
            out2 = tf.nn.relu(out2)
        
            out2_t = tf.nn.conv2d_transpose(out2, kernel * 4, tf.shape(pyramid[2]), [1, 2, 2, 1])

        # subnet 3
        with tf.variable_scope('subnet3'):
            out3 = subnet(pyramid[2],num_channels, int((num_feature) / 4))
            out3 = tf.add(out3, out2_t)
            out3 = tf.nn.relu(out3)
            #out3=lrelu(out3)
          
            out3_t = tf.nn.conv2d_transpose(out3, kernel * 4, tf.shape(pyramid[3]), [1, 2, 2, 1])

        # subnet 4
        with tf.variable_scope('subnet4'):
            out4 = subnet(pyramid[3], num_channels,int((num_feature) / 2))
            out4 = tf.add(out4, out3_t)
            out4 = tf.nn.relu(out4)
            #out4=lrelu(out4)
            #out4_t = tf.nn.conv2d_transpose(out4, kernel * 4, tf.shape(pyramid[4]), [1, 2, 2, 1])

        """    
        #subnet 5
        with tf.variable_scope('subnet5'):
            out5 = subnet(pyramid[4], num_channels,int(num_feature))
            out5 = tf.add(out5, out4_t)
            #out5=lrelu(out5)
            out5 = tf.nn.relu(out5)
        """
        outout_pyramid = []
        outout_pyramid.append(out1)
        outout_pyramid.append(out2)
        outout_pyramid.append(out3)
        outout_pyramid.append(out4)
        #outout_pyramid.append(out5)

        return outout_pyramid




if __name__ == '__main__':
    tf.reset_default_graph()   
    
    input_x = tf.placeholder(tf.float32, [10,None,None,3])
    
    outout_pyramid  = inference(input_x)
    var_list = tf.trainable_variables()   
    print("Total parameters' number: %d" 
         %(np.sum([np.prod(v.get_shape().as_list()) for v in var_list])))  
