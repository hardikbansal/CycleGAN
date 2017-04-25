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

img_height = 128
img_width = 128
img_layer = 3
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = False
output_path = "output"
check_dir = "/output/checkpoints/"


temp_check = 0



max_epoch = 1
max_images = 1000

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
ngf = 64
ndf = 64



def build_resnet_block(inputres, dim, name="resnet"):
    out_res = inputres
    with tf.variable_scope(name):
        out_res = general_conv2d(inputres, dim, 3, 3, 1, 1, 0.02, "SAME","c1")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "SAME","c2",do_relu=False)

        out_res = tf.nn.relu(out_res + inputres,"relu")
    return out_res


def build_generator_resnet_6blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        
        pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02,name="c1")
        o_c2 = general_conv2d(o_c1, ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
        o_c3 = general_conv2d(o_c2, ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

        o_r1 = build_resnet_block(o_c3, ngf*4, "r1")
        o_r2 = build_resnet_block(o_r1, ngf*4, "r2")
        o_r3 = build_resnet_block(o_r2, ngf*4, "r3")
        o_r4 = build_resnet_block(o_r3, ngf*4, "r4")
        o_r5 = build_resnet_block(o_r4, ngf*4, "r5")
        o_r6 = build_resnet_block(o_r5, ngf*4, "r6")

        o_c4 = general_deconv2d(o_r6, [batch_size,64,64,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
        o_c5 = general_deconv2d(o_c4, [batch_size,128,128,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
        o_c6 = general_conv2d(o_c5, img_layer, f, f, 1, 1, 0.02,"SAME","c6",do_relu="False")

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6,"t1")


    return out_gen

def build_generator_resnet_9blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        
        pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02,name="c1")
        o_c2 = general_conv2d(o_c1, ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
        o_c3 = general_conv2d(o_c2, ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

        o_r1 = build_resnet_block(o_c3, ngf*4, "r1")
        o_r2 = build_resnet_block(o_r1, ngf*4, "r2")
        o_r3 = build_resnet_block(o_r2, ngf*4, "r3")
        o_r4 = build_resnet_block(o_r3, ngf*4, "r4")
        o_r5 = build_resnet_block(o_r4, ngf*4, "r5")
        o_r6 = build_resnet_block(o_r5, ngf*4, "r6")
        o_r7 = build_resnet_block(o_r6, ngf*4, "r7")
        o_r8 = build_resnet_block(o_r7, ngf*4, "r8")
        o_r9 = build_resnet_block(o_r8, ngf*4, "r9")

        o_c4 = general_deconv2d(o_r9, [batch_size,128,128,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
        o_c5 = general_deconv2d(o_c4, [batch_size,256,256,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
        o_c6 = general_conv2d(o_c5, img_layer, f, f, 1, 1, 0.02,"SAME","c6",do_relu="False")

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6,"t1")


    return out_gen


def build_gen_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

    return o_c5



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
    image_A = tf.subtract(tf.div(tf.image.resize_image_with_crop_or_pad(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[143,143]), 128, 128),127.5),1)
    image_B = tf.subtract(tf.div(tf.image.resize_image_with_crop_or_pad(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[143,143]), 128, 128),127.5),1)

    


    #Build the network

    input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
    input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
    
    fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
    fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")

    global_step = tf.Variable(0, name="global_step", trainable=False)

    num_fake_inputs = 0

    lr = tf.placeholder(tf.float32, shape=[], name="lr")

    with tf.variable_scope("Model") as scope:
        fake_B = build_generator_resnet_6blocks(input_A, name="g_A")
        fake_A = build_generator_resnet_6blocks(input_B, name="g_B")
        rec_A = build_gen_discriminator(input_A, "d_A")
        rec_B = build_gen_discriminator(input_B, "d_B")

        scope.reuse_variables()

        fake_rec_A = build_gen_discriminator(fake_A, "d_A")
        fake_rec_B = build_gen_discriminator(fake_B, "d_B")
        cyc_A = build_generator_resnet_6blocks(fake_B, "g_B")
        cyc_B = build_generator_resnet_6blocks(fake_A, "g_A")

        scope.reuse_variables()

        fake_pool_rec_A = build_gen_discriminator(fake_pool_A, "d_A")
        fake_pool_rec_B = build_gen_discriminator(fake_pool_B, "d_B")


    # Loss functions for various things

    cyc_loss = tf.reduce_mean(tf.abs(input_A-cyc_A)) + tf.reduce_mean(tf.abs(input_B-cyc_B))
    

    disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_rec_A,1))
    disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_rec_B,1))
    
    g_loss_A = cyc_loss*10 + disc_loss_B
    g_loss_B = cyc_loss*10 + disc_loss_A

    d_loss_A = tf.reduce_mean(tf.square(rec_A)) + tf.reduce_mean(tf.squared_difference(fake_pool_rec_A,1))
    d_loss_B = tf.reduce_mean(tf.square(rec_B)) + tf.reduce_mean(tf.squared_difference(fake_pool_rec_B,1))

    
    optimizer = tf.train.AdamOptimizer(lr)

    model_vars = tf.trainable_variables()

    d_A_vars = [var for var in model_vars if 'd_A' in var.name]
    g_A_vars = [var for var in model_vars if 'g_A' in var.name]
    d_B_vars = [var for var in model_vars if 'd_B' in var.name]
    g_B_vars = [var for var in model_vars if 'g_B' in var.name]
    
    d_A_trainer = optimizer.minimize(-d_loss_A, var_list=d_A_vars)
    d_B_trainer = optimizer.minimize(-d_loss_B, var_list=d_B_vars)
    g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
    g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)


    # Summary Variables

    tf.summary.scalar("g_A_loss", g_loss_A)
    tf.summary.scalar("g_B_loss", g_loss_B)
    tf.summary.scalar("d_A_loss", d_loss_A)
    tf.summary.scalar("d_B_loss", d_loss_B)

    summary_op = tf.summary.merge_all()
    


    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    

    with tf.Session() as sess:
        sess.run(init)


        if to_restore:
            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)


        if to_test:
            print "Testing the results"


            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            num_files_A = sess.run(queue_length_A)
            num_files_B = sess.run(queue_length_B)

            images_A = []
            images_B = []

            fake_images_A = np.zeros((pool_size,img_height, img_width, img_layer))
            fake_images_B = np.zeros((pool_size,img_height, img_width, img_layer))

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

            fake_images_A = np.zeros((pool_size,img_height, img_width, img_layer))
            fake_images_B = np.zeros((pool_size,img_height, img_width, img_layer))

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

            # Traingin Loop

            writer = tf.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # a,b,c,d,e = sess.run([cyc_loss,disc_loss_A,disc_loss_B,g_loss_A,g_loss_B],feed_dict={input_A:A_input[0], input_B:B_input[0], fake_pool_A:fake_images_A, fake_pool_B:fake_images_B})
            # print(a,b,c,d,e)

            for epoch in range(sess.run(global_step),10):
                print ("In the epoch ", epoch)

                saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)


                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100

                summary_str, cyc_A_temp = sess.run([summary_op, cyc_A],feed_dict={input_A:A_input[0], input_B:B_input[0], fake_pool_A:fake_images_A, fake_pool_B:fake_images_B})
                imsave("./output/output_"+str(epoch)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/input.jpg",((A_input[0][0]+1)*127.5).astype(np.uint8))


                
                writer.add_summary(summary_str, epoch)




                for ptr in range(0,max_images):

                    print("In the iteration ",ptr)

                    print(time.time()*1000.0)

                    if(num_fake_inputs < pool_size):
                    
                        _, fake_B_temp = sess.run([g_A_trainer,fake_B],feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr})
                        sess.run(d_B_trainer,feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr, fake_pool_B:fake_images_B[0:num_fake_inputs+1]})
                        
                        _, fake_A_temp = sess.run([g_B_trainer, fake_A],feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr})
                        sess.run(d_A_trainer,feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr, fake_pool_A:fake_images_A[0:num_fake_inputs+1]})
                
                        # summary_str = sess.run(summary_op,feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], fake_pool_A:fake_images_A, fake_pool_B:fake_images_B})
                    else :
                        _, fake_B_temp = sess.run([g_A_trainer,fake_B],feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr})
                        sess.run(d_B_trainer,feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr, fake_pool_B:fake_images_B})
                        
                        _, fake_A_temp = sess.run([g_B_trainer, fake_A],feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr})
                        sess.run(d_A_trainer,feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], lr:curr_lr, fake_pool_A:fake_images_A})

                        # summary_str = sess.run(summary_op,feed_dict={input_A:A_input[ptr], input_B:B_input[ptr], fake_pool_A:fake_images_A, fake_pool_B:fake_images_B})
                    

                    # writer.add_summary(summary_str, epoch*max_images + ptr)

                    if(num_fake_inputs < pool_size):
                        fake_images_A[num_fake_inputs] = fake_A_temp[0]
                        fake_images_B[num_fake_inputs] = fake_B_temp[0]
                        num_fake_inputs+=1
                    else :
                        p = random.random()
                        if p > 0.5:
                            random_id = random.randint(0,pool_size-1)
                            fake_images_A[random_id] = fake_A_temp[0]
                            random_id = random.randint(0,pool_size-1)
                            fake_images_B[random_id] = fake_B_temp[0]

                sess.run(tf.assign(global_step, epoch + 1))

                    


                # if(i % 10 == 0):
                #     saver.save(sess,"/output/cyleganmodel")

            writer.add_graph(sess.graph)













# def train():
#     mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#     x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
#     z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
#     keep_prob = tf.placeholder(tf.float32, name="keep_prob")
#     global_step = tf.Variable(0, name="global_step", trainable=False)

#     x_generated, g_params = build_generator(z_prior)
#     y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

#     d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
#     g_loss = - tf.log(y_generated)

#     optimizer = tf.train.AdamOptimizer(0.0001)

#     d_trainer = optimizer.minimize(d_loss, var_list=d_params)
#     g_trainer = optimizer.minimize(g_loss, var_list=g_params)

#     init = tf.initialize_all_variables()

#     saver = tf.train.Saver()

#     sess = tf.Session()

#     sess.run(init)

    # if to_restore:
    #     chkpt_fname = tf.train.latest_checkpoint(output_path)
    #     saver.restore(sess, chkpt_fname)
    # else:
    #     if os.path.exists(output_path):
    #         shutil.rmtree(output_path)
    #     os.mkdir(output_path)


#     z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

#     for i in range(sess.run(global_step), max_epoch):
#         for j in range(60000 / batch_size):
#             print "epoch:%s, iter:%s" % (i, j)
#             x_value, _ = mnist.train.next_batch(batch_size)
#             x_value = 2 * x_value.astype(np.float32) - 1
#             z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
#             sess.run(d_trainer,
#                      feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
#             if j % 1 == 0:
#                 sess.run(g_trainer,
#                          feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
#         x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
#         show_result(x_gen_val, "output/sample{0}.jpg".format(i))
#         z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
#         x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
#         show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
#         sess.run(tf.assign(global_step, i + 1))
#         saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)

def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)

def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, "output/test_result.jpg")


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()