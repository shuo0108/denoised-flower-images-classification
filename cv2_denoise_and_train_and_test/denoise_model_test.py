#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:48:32 2018

@author: hs
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image  
import os
import tensorflow as tf
import numpy as np
import cv2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#label_lines = [line.rstrip() for line in tf.gfile.GFile("model/output_labels.txt")]
def read_and_decode(tfrecord_file_path):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([tfrecord_file_path],shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'data':tf.FixedLenFeature([256*256],tf.float32),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'id' : tf.FixedLenFeature([], tf.int64),
                                       })
    image=features['data']
    image=tf.reshape(image,[256,256])
    label = tf.cast(features['label'], tf.int64)
    num= tf.cast(features['id'], tf.int64)
    return image,label,num

def image_made(tfrecord_path,decord_img_path):
    new_folder_path=decord_img_path
    
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    image,label,num=read_and_decode(tfrecord_path)   
    im_num=0
    for record in tf.python_io.tf_record_iterator(tfrecord_path):
       im_num += 1 
    print('num of text image:'+str(im_num))
    
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        #启动多线程
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。       
        L_all=[]
        for i in range(im_num):
            a,b,c=sess.run([image,label,num])
            img=(a+1)*128
            im=Image.fromarray(img).convert('L')
            im.save(new_folder_path+'/'+str(c)+'_''Label_'+str(b)+'.jpeg')#存下图片
            L_all.append(b)

            
        coord.request_stop()
        coord.join(threads)
    return L_all



def load_graph(frozen_graph_filename):  
# We parse the graph_def file  
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
      
    # We load the graph_def in the default graph  
    with tf.Graph().as_default() as graph:  
        tf.import_graph_def(  
            graph_def,   
            input_map=None,   
            return_elements=None,  
            name='DNN',
            op_dict=None,   
            producer_op_list=None  
        )  
    return graph 

def denoise(image_dir,denoise_im_dir):
    if not os.path.exists(denoise_im_dir):
        os.mkdir(denoise_im_dir)

    #加载已经将参数固化后的图  
    graph = load_graph(pb_name)  
  
    # We can list operations  
    #op.values() gives you a list of tensors it produces  
    #op.name gives you the name  
    #输入,输出结点也是operation,所以,我们可以得到operation的名字  

        # prefix/Placeholder/inputs_placeholder  
        # ...  
        # prefix/Accuracy/predictions  
    #操作有:prefix/Placeholder/inputs_placeholder  
    #操作有:prefix/Accuracy/predictions  
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字  
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字  
    y = graph.get_tensor_by_name('DNN/final_output:0') 
    x = graph.get_tensor_by_name('DNN/noise_image:0')  
    is_traing=graph.get_tensor_by_name('DNN/is_training:0')
     
          
    with tf.Session(graph=graph) as sess:  
        image_path=os.listdir(image_dir)
        for im in image_path:
            noise_image= load_images(os.path.join(image_dir,im)).astype(np.float32) / 255.0
           
            #output_clean_image,noisy_image = sess.run([y,x],feed_dict={x:noise_image})
            
            output_clean_image,noisy_image = sess.run([y,x],feed_dict={x:noise_image,is_traing:False})
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            
            save_images(os.path.join(denoise_im_dir,im), outputimage)
            print(im+' has been denoised and saved')
            
       # [[ 0.]] Yay!  
    print ("finish")  
    



def psnr(A, B):
    return 10*np.log(255*255.0/(((A.astype(np.float)-B)**2).mean()))/np.log(10)

def double2uint8(I, ratio=1.0):
    return np.clip(np.round(I*ratio), 0, 255).astype(np.uint8)

def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))
    return kernel/kernel.sum()

def NLmeansfilter(I, h_=10, templateWindowSize=5,  searchWindowSize=11):
    f = int(templateWindowSize/2)
    t = int(searchWindowSize/2)
    height, width = I.shape[:2]
    padLength = t+f
    I2 = np.pad(I, int(padLength), 'symmetric')
    kernel = make_kernel(int(f))
    h = (h_**2)
    I_ = I2[int(padLength-f):int(padLength+f+height), int(padLength-f):int(padLength+f+width)]

    average = np.zeros(I.shape)
    sweight = np.zeros(I.shape)
    wmax =  np.zeros(I.shape)
    for i in range(-t, t+1):
        for j in range(-t, t+1):
            if i==0 and j==0:
                continue
            I2_ = I2[int(padLength+i-f):int(padLength+i+f+height), int(padLength+j-f):int(padLength+j+f+width)]
            w = np.exp(-cv2.filter2D((I2_ - I_)**2, -1, kernel)/h)[f:f+height, f:f+width]
            sweight += w
            wmax = np.maximum(wmax, w)
            average += (w*I2_[f:f+height, f:f+width])
    return (average+wmax*I)/(sweight+wmax)
def im_denoise(im_dir,sigma,de_im_dir):
    im_names=os.listdir(im_dir)
    for im in im_names:
        img=cv2.imread(os.path.join(im_dir,im),0)
        img_ = double2uint8(NLmeansfilter(img.astype(np.float), sigma, 5, 11))
        de_im=Image.fromarray(img_).convert('L')
        de_im.save(os.path.join(de_im_dir,im))
        print(im+' has been denoised and saved')


def model_text(tfrecord_path):
    tf.reset_default_graph()
    img_dir='test_hs_1'
    denoise_imgdir_path='test_hs_denoise'

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
    if not os.path.exists(denoise_imgdir_path):
        os.makedirs(denoise_imgdir_path)
    #从tfrecord文件中解出图片放在img_dir
    L_all=image_made(tfrecord_path,img_dir)
    
    #对于解出的图片去噪放在denoise_imgdir_path
    #denoise(img_dir,denoise_imgdir_path)
    
    im_denoise(img_dir,25,denoise_imgdir_path)
    denoise_img_path=os.listdir(denoise_imgdir_path)
    path=denoise_imgdir_path
  


    

    # Unpersists graph from file
    with tf.gfile.FastGFile('denoise_test.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        w=0
        num_s=[]
        L=[]
        L_true=[]
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        for im_p in denoise_img_path:
            im_path=os.path.join(path,im_p)
            image_data = tf.gfile.FastGFile(im_path, 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})   
            label_p=tf.arg_max(predictions,1)
            label_p=sess.run(label_p)
            l=int(label_p)+1
            num=int(im_p.split('/')[-1].split('.')[0].split('_')[0])
            num_s.append(num)
            L.append(l)
           
            b=im_p.split('/')[-1].split('.')[0].split('_')[-1]

            L_true.append(int(b))
            if int(l)!=int(b):
                print(im_p+' is wrong!  '+'wrong label:'+str(l)+'   true label:'+str(b))
                w+=1
        n_all=[(num_s[i],L[i],L_true[i]) for i in range(len(L))]
        n_all.sort(key=lambda x:x[0])
        L=[n_all[i][1] for i in range(len(num_s))]
        L_true=[n_all[i][2] for i in range(len(num_s))]

    return L#,L_all

def main():
    
    label=model_text('TFcordX_text.tfrecord')
    '''
    tfrecord_path='/home/hs/flowers_classification_1/data/TFcodeX_10.tfrecord'
    
    label,l=model_text(tfrecord_path)
    
    
    correct_prediction=tf.equal(label,l)
    accuary=tf.reduce_mean(tf.cast(correct_prediction,'float'))
    with tf.Session() as sess:
        a=sess.run(accuary)
        print('accuary is:'+str(a))
    '''
    
    return label
    

if __name__ == '__main__':
    main()

    
    




  

