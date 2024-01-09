#!/usr/bin/env python2
# -*- coding: utf-8 -*-




import scipy
import os
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from model_new import *


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_pyramids = 5       # number of pyramid levels
learning_rate = 0.001   # learning rate
iterations = int(1e4)  # iterations
batch_size = 10       # batch size
num_channels = 3       # number of input's channels 
patch_size = 64      # patch size
save_model_path = './model'  # path of saved model
model_name = 'model-epoch'    # name of saved model


input_path = './TrainData/data/' # underwater images

gt_path = './TrainData/gt/'    # ground truth


def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

vgg_path=scipy.io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')
print("[i] Loaded pre-trained vgg19 parameters")
# build VGG19 to load pre-trained parameters
def build_vgg19(input,reuse=False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_path['layers'][0]
        net['input']=input
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
        net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
        return net

def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input-output))

def compute_percep_loss(input, output, reuse=False):
    vgg_real=build_vgg19(output,reuse=reuse)
    vgg_fake=build_vgg19(input,reuse=True)
    p0=compute_l1_loss(vgg_real['input'],vgg_fake['input'])
    p1=compute_l1_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])/2.6
    p2=compute_l1_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])/4.8
    p3=compute_l1_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])/3.7
    p4=compute_l1_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])/5.6
    p5=compute_l1_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10/1.5
    p=p0+p1+p2+p3+p4+p5
    return p










# randomly select image patches
def _parse_function(input_path, gt_path, patch_size = patch_size):   
    image_string = tf.read_file(input_path)  
    image_decoded = tf.image.decode_png(image_string, channels=3)  
    rainy = tf.cast(image_decoded, tf.float32)/255.0
          
    image_string = tf.read_file(gt_path)  
    image_decoded = tf.image.decode_png(image_string, channels=3)  
    label = tf.cast(image_decoded, tf.float32)/255.0
          
    t = time.time()
    Data = tf.random_crop(rainy, [patch_size, patch_size ,3],seed = t)   # randomly select patch
    Label = tf.random_crop(label, [patch_size, patch_size ,3],seed = t)       
    return Data, Label 



if __name__ == '__main__':    
    tf.reset_default_graph()
    input_files = os.listdir(input_path)
    #print(len(input_files))
    for i in range(len(input_files)):
        input_files[i] = input_path + input_files[i]
        
    label_files = os.listdir(gt_path)       
    for i in range(len(label_files)):
        label_files[i] = gt_path + label_files[i] 
    
    input_files = tf.convert_to_tensor(input_files, dtype=tf.string)  
    label_files = tf.convert_to_tensor(label_files, dtype=tf.string)  

    dataset = tf.data.Dataset.from_tensor_slices((input_files, label_files))
    dataset = dataset.map(_parse_function)    
    dataset = dataset.prefetch(buffer_size=batch_size * 10)
    dataset = dataset.batch(batch_size).repeat()  
    iterator = dataset.make_one_shot_iterator()   
    inputs, labels = iterator.get_next()  


    k = np.float32([.0625, .25, .375, .25, .0625])  # Gaussian kernel for image pyramid
    k = np.outer(k, k) 
    kernel = k[:,:,None,None]/k.sum()*np.eye(3, dtype = np.float32)
    labels_GaussianPyramid = GaussianPyramid( labels, kernel, (num_pyramids-1) ) # Gaussian pyramid for ground truth

    outout_pyramid = inference(inputs) # LPNet
    
#    loss_ssim_train1 = loss_ssim(outout_pyramid[0], labels_GaussianPyramid[0], batch_size, num_channels)
   # loss_ssim_train2 = loss_ssim(outout_pyramid[1], labels_GaussianPyramid[1], batch_size, num_channels)
    #loss_ssim_train3 = loss_ssim(outout_pyramid[2], labels_GaussianPyramid[2], batch_size, num_channels)
   # loss_ssim_train4 = loss_ssim(outout_pyramid[3], labels_GaussianPyramid[3], batch_size, num_channels)
    #loss_ssim_train5 = loss_ssim(outout_pyramid[4], labels, batch_size, num_channels)
    loss1 = tf.reduce_mean(tf.abs(outout_pyramid[0] - labels_GaussianPyramid[0]))    # L1 loss
    loss11=compute_percep_loss(outout_pyramid[0],labels_GaussianPyramid[0]) # PERCEP_LOSS

    loss2 = tf.reduce_mean(tf.abs(outout_pyramid[1] - labels_GaussianPyramid[1]))    # L1 loss
    loss21 = compute_percep_loss(outout_pyramid[1], labels_GaussianPyramid[1])  # PERCEP_LOSS

    loss3 = tf.reduce_mean(tf.abs(outout_pyramid[2] - labels_GaussianPyramid[2]))    # L1 loss
    loss31 = compute_percep_loss(outout_pyramid[2], labels_GaussianPyramid[2])  

    loss4 = tf.reduce_mean(tf.abs(outout_pyramid[3] - labels_GaussianPyramid[3]))   # L1 loss
    loss42 = tf.reduce_mean((1. - tf.image.ssim(outout_pyramid[3],labels_GaussianPyramid[3], max_val=1.0))/2.) # SSIM loss
    loss41=compute_percep_loss(outout_pyramid[3], labels_GaussianPyramid[3])

    loss5 = tf.reduce_mean(tf.abs(outout_pyramid[4] - labels))  # L1 loss
    loss52 = tf.reduce_mean((1. - tf.image.ssim(outout_pyramid[4],labels, max_val=1.0))/2.) # SSIM loss
    loss51 = compute_percep_loss(outout_pyramid[4], labels)

    #loss = loss1+loss2+loss3+loss4+loss5+(loss11+loss21+loss31+loss41+loss51)*0.02+loss42+loss52
    #loss = loss2+loss21*0.02 + loss3+loss31*0.02 +loss4+ loss41*0.02 + loss5 + loss51*0.02
    loss = loss1+loss2+loss3+loss4+loss5+loss42+loss52
 




    g_optim =  tf.train.AdamOptimizer(learning_rate).minimize(loss) # Optimization method: Adam
    
    all_vars = tf.trainable_variables()
    saver = tf.train.Saver(var_list=all_vars, max_to_keep = 5)
   

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
       
       sess.run(tf.group(tf.global_variables_initializer(), 
                         tf.local_variables_initializer()))
       tf.get_default_graph().finalize()	
              
       if tf.train.get_checkpoint_state(save_model_path):   # load previous trained model 
          ckpt = tf.train.latest_checkpoint(save_model_path)
          saver.restore(sess, ckpt)  
          ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
          start_point = int(ckpt_num[0]) + 1     
          print("loaded successfully")
       else:  # re-training when no models found
          print("re-training")
          start_point = 0  
          
       check_input, check_label =  sess.run([inputs,labels])
       print("check patch pair:")  
       plt.subplot(1,3,1)     
       plt.imshow(check_input[0,:,:,:])
       plt.title('input')         
       plt.subplot(1,3,2)    
       plt.imshow(check_label[0,:,:,:])
       plt.title('ground truth')      
       plt.show()    
     
       start = time.time()    
       
       for j in range(start_point,iterations):                   
                          
           _, Training_Loss = sess.run([g_optim,loss])  # training

           if np.mod(j+1,10) == 0 and j != 0:
              end = time.time() 
              print ('%d / %d iteraions, Training Loss  = %.4f, running time = %.1f s' 
                     % (j+1, iterations, Training_Loss, (end-start)))          
              save_path_full = os.path.join(save_model_path, model_name) 
              saver.save(sess, save_path_full, global_step = j+1) # save model every 100 iterations
              start = time.time()  
              
       print ('training finished') 
    sess.close()
