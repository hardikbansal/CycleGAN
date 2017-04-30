# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random


from layers import *
from model import *


img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = True
output_path = "output"
check_dir = "./output/checkpoints/"


temp_check = 0



max_epoch = 1
max_images = 1000

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
ngf = 32
ndf = 64



def fake_image_pool(num_fakes, fake, fake_pool):

    if(num_fakes < pool_size):
        fake_pool[num_fakes] = fake
        return fake
    else :
        p = random.random()
        if p > 0.5:
            random_id = random.randint(0,pool_size-1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = fake
            return temp
        else :
            return fake




def train():


    # Load Dataset from the dataset folder

    filenames_A = tf.train.match_filenames_once("./input/horse2zebra/trainA/*.jpg")    
    queue_length_A = tf.size(filenames_A)
    filenames_B = tf.train.match_filenames_once("./input/horse2zebra/trainB/*.jpg")    
    queue_length_B = tf.size(filenames_B)
    
    filename_queue_A = tf.train.string_input_producer(filenames_A)
    filename_queue_B = tf.train.string_input_producer(filenames_B)
    
    image_reader = tf.WholeFileReader()
    _, image_file_A = image_reader.read(filename_queue_A)
    _, image_file_B = image_reader.read(filename_queue_B)
    image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[256,256]),127.5),1)
    image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[256,256]),127.5),1)

    print(tf.__version__)

    


    #Build the network

    input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
    input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
    
    fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
    fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")

    global_step = tf.Variable(0, name="global_step", trainable=False)

    num_fake_inputs = 0

    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    with tf.variable_scope("Model") as scope:
        fake_B = build_generator_resnet_9blocks(input_A, name="g_A")
        fake_A = build_generator_resnet_9blocks(input_B, name="g_B")
        rec_A = build_gen_discriminator(input_A, "d_A")
        rec_B = build_gen_discriminator(input_B, "d_B")

        scope.reuse_variables()

        fake_rec_A = build_gen_discriminator(fake_A, "d_A")
        fake_rec_B = build_gen_discriminator(fake_B, "d_B")
        cyc_A = build_generator_resnet_9blocks(fake_B, "g_B")
        cyc_B = build_generator_resnet_9blocks(fake_A, "g_A")

        scope.reuse_variables()

        fake_pool_rec_A = build_gen_discriminator(fake_pool_A, "d_A")
        fake_pool_rec_B = build_gen_discriminator(fake_pool_B, "d_B")


    # Loss functions for various things

    cyc_loss = tf.reduce_mean(tf.abs(input_A-cyc_A)) + tf.reduce_mean(tf.abs(input_B-cyc_B))
    

    disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_rec_A,1))
    disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_rec_B,1))
    
    g_loss_A = cyc_loss*10 + disc_loss_B
    g_loss_B = cyc_loss*10 + disc_loss_A

    d_loss_A = (tf.reduce_mean(tf.square(fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(rec_A,1)))/2.0
    d_loss_B = (tf.reduce_mean(tf.square(fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(rec_B,1)))/2.0

    
    optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)

    model_vars = tf.trainable_variables()

    d_A_vars = [var for var in model_vars if 'd_A' in var.name]
    g_A_vars = [var for var in model_vars if 'g_A' in var.name]
    d_B_vars = [var for var in model_vars if 'd_B' in var.name]
    g_B_vars = [var for var in model_vars if 'g_B' in var.name]
    
    d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
    d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
    g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
    g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

    for var in model_vars: print(var.name)




    # Summary Variables

    g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
    g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
    d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
    d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    # summary_op = tf.summary.merge_all()
    


    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    

    with tf.Session() as sess:
        sess.run(init)


        if to_restore:
            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

        if to_test:
            print("Testing the results")


            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            num_files_A = sess.run(queue_length_A)
            num_files_B = sess.run(queue_length_B)

            images_A = []
            images_B = []

            A_input = np.zeros((max_images,batch_size,img_height, img_width, img_layer))
            B_input = np.zeros((max_images,batch_size,img_height, img_width, img_layer))

            for i in range(max_images): 
                image_tensor = sess.run(image_A)
                A_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

            for i in range(max_images):
                image_tensor = sess.run(image_B)
                B_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))


            coord.request_stop()
            coord.join(threads)

            for ptr in range(0,100):
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([fake_A, fake_B, cyc_A, cyc_B],feed_dict={input_A:A_input[0], input_B:B_input[0]})
                imsave("./output/fakeB_"+str(ptr)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/fakeA_"+str(ptr)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/cycA_"+str(ptr)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/cycB_"+str(ptr)+".jpg",((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/inputA_"+str(ptr)+".jpg",((A_input[0][0]+1)*127.5).astype(np.uint8))
                imsave("./output/inputB_"+str(ptr)+".jpg",((B_input[0][0]+1)*127.5).astype(np.uint8))


        else :

            # Loading images into the tensors
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            num_files_A = sess.run(queue_length_A)
            num_files_B = sess.run(queue_length_B)

            images_A = []
            images_B = []

            fake_images_A = np.zeros((pool_size,1,img_height, img_width, img_layer))
            fake_images_B = np.zeros((pool_size,1,img_height, img_width, img_layer))


            A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
            B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))

            for i in range(max_images): 
                image_tensor = sess.run(image_A)
                A_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

            for i in range(max_images):
                image_tensor = sess.run(image_B)
                B_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))


            coord.request_stop()
            coord.join(threads)

            # Traingin Loop

            writer = tf.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)


            for epoch in range(sess.run(global_step),100):
                print ("In the epoch ", epoch)

                saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)


                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100


                for i in range(0,10):
                    fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([fake_A, fake_B, cyc_A, cyc_B],feed_dict={input_A:A_input[i], input_B:B_input[i]})
                    imsave("/output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                    imsave("/output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                    imsave("/output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
                    imsave("/output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
                    imsave("/output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",((A_input[i][0]+1)*127.5).astype(np.uint8))
                    imsave("/output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",((B_input[i][0]+1)*127.5).astype(np.uint8))





                for ptr in range(0,max_images):

                    print("In the iteration ",ptr)

                    print("Starting",time.time()*1000.0)

                    # Optimizing the G_A network

                    _, fake_B_temp, summary_str = sess.run([g_A_trainer, fake_B, g_A_loss_summ],feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    print("After gA", time.time()*1000.0)
                    
                    fake_B_temp1 = fake_image_pool(num_fake_inputs, fake_B_temp, fake_images_B)
                    
                    # Optimizing the D_B network
                    _, summary_str = sess.run([d_B_trainer, d_B_loss_summ],feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr, fake_pool_B:fake_B_temp1})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    print("After dB", time.time()*1000.0)
                    
                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run([g_B_trainer, fake_A, g_B_loss_summ],feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    print("After gB", time.time()*1000.0)
                    
                    fake_A_temp1 = fake_image_pool(num_fake_inputs, fake_A_temp, fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run([d_A_trainer, d_A_loss_summ],feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr, fake_pool_A:fake_A_temp1})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    print("After dA", time.time()*1000.0)

                    num_fake_inputs+=1
            
                        

                sess.run(tf.assign(global_step, epoch + 1))

            writer.add_graph(sess.graph)



if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()