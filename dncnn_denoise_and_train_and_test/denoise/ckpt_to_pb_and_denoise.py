#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:50:48 2018

@author: hs
"""


from glob import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util  
from tensorflow.python import pywrap_tensorflow

from model import denoiser
from utils import *



tf.reset_default_graph()
ckpt_dir='/home/cpssface/hs/flowers_classification_1/denoise/checkpoint_final'
pb_name='denoise.pb'
save_dir='test'



def get_ckpt_node_name(ckpt_dir):

    ckpt=tf.train.get_checkpoint_state(ckpt_dir)
    print(ckpt)
    print('---------------------------------')
    ckpt_path=ckpt.model_checkpoint_path
    print(ckpt_path)
    print('---------------------------------')
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map() 
    
    for key in var_to_shape_map: 
        print("tensor_name: ", key)
        print('---------------------------------')



#get_ckpt_node_name()


def ckpt_pb(checkpoint_dir,output_graph):
    print("[*] Reading checkpoint...")
    
    
    checkpoint= tf.train.get_checkpoint_state(checkpoint_dir)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    output_node_names = "final_output" 
    clear_devices = True  
      
    # We import the meta graph and retrive a Saver  
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)  
  
    # We retrieve the protobuf graph definition  
    graph = tf.get_default_graph()  
    input_graph_def = graph.as_graph_def()  
  
    #We start a session and restore the graph weights  
    #这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen  
    #相当于将参数已经固化在了图当中   
    with tf.Session() as sess:  
        saver.restore(sess, input_checkpoint)  
  
        # We use a built-in TF helper to export variables to constant  
        output_graph_def = graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names.split(",") # We split on comma for convenience  
        )   
  
        # Finally we serialize and dump the output graph to the filesystem  
        with tf.gfile.GFile(output_graph, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
        print("%d ops in the final graph." % len(output_graph_def.node))  
    
 

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
    
    for i,op in enumerate (graph.get_operations()):  
        if i in range(10):
            print(op.name,op.values())  
            print('----------------')
      
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


if __name__ =='__main__':
    denoise('/home/cpssface/hs/flowers_classification_1/denoise/data/train/train_c','test1')
    #ckpt_pb(ckpt_dir,'denoise_final.pb')
