# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil
from PIL import Image


from layers import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = "output"



max_epoch = 1

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
sample_size = 10
ngf = 128
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
        
        o_c1 = general_conv2d(inputgen, ngf, f, f, 1, 1, 0.02,"SAME","c1")
        o_c2 = general_conv2d(o_c1, ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
        o_c3 = general_conv2d(o_c2, ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

        o_r1 = build_resnet_block(o_c3, ngf*4, "r1")
        o_r2 = build_resnet_block(o_r1, ngf*4, "r2")
        o_r3 = build_resnet_block(o_r2, ngf*4, "r3")
        o_r4 = build_resnet_block(o_r3, ngf*4, "r4")
        o_r5 = build_resnet_block(o_r4, ngf*4, "r5")
        o_r6 = build_resnet_block(o_r5, ngf*4, "r6")

        o_c4 = general_deconv2d(o_r6, [batch_size,128,128,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
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

        out_disc = tf.nn.sigmoid(o_c5,"sigmoid")

    return out_disc



def train():


    # Load Dataset from the dataset folder

    filenames_A = tf.train.match_filenames_once("../datasets/horse2zebra/trainA/*.jpg")    
    queue_length_A = tf.size(filenames_A)
    filenames_B = tf.train.match_filenames_once("../datasets/horse2zebra/trainB/*.jpg")    
    queue_length_B = tf.size(filenames_B)
    
    filename_queue_A = tf.train.string_input_producer(filenames_A)
    filename_queue_B = tf.train.string_input_producer(filenames_B)
    
    image_reader = tf.WholeFileReader()
    _, image_file_A = image_reader.read(filename_queue_A)
    _, image_file_B = image_reader.read(filename_queue_B)
    image_A = tf.image.decode_jpeg(image_file_A)
    image_B = tf.image.decode_jpeg(image_file_B)

    


    #Build the network

    input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
    # input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")

    # fake_A = build_generator_resnet_6blocks(input_A, name="d_A")
    # fake_B = build_generator_resnet_6blocks(input_B, name="d_B")
    rec_A = build_generator_resnet_6blocks(input_A, "d_A")


    d_loss = tf.reduce_sum(rec_A)

    optimizer = tf.train.AdamOptimizer(0.0001)

    d_trainer = optimizer.minimize(d_loss, var_list=tf.trainable_variables())

    init = tf.global_variables_initializer()

    

    with tf.Session() as sess:
        sess.run(init)


        # Loading images into the tensors
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_files_A = sess.run(queue_length_A)
        num_files_B = sess.run(queue_length_B)

        images_A = []
        images_B = []

        for i in range(10):
            image_tensor = sess.run(image_A)
            images_A.append(image_tensor)

        for i in range(10):
            image_tensor = sess.run(image_B)
            images_B.append(image_tensor)

        # Image.fromarray(np.asarray(image_tensor)).save("testimg.jpg")

        Train_A = tf.stack(images_A)
        Train_B = tf.stack(images_B)

        num_images = Train_B.shape

        print(num_images)

        coord.request_stop()
        coord.join(threads)

        # Traingin Loop

        writer = tf.summary.FileWriter("output/1")

        for i in range(0,1):
            for j in range(0,3):
                print("next iter")
                A_input = images_A[1]
                A_input = tf.reshape(A_input,[batch_size,img_height, img_width, img_layer])
                sess.run(d_trainer,feed_dict={input_A:A_input.eval(session=sess)})

        writer.add_graph(sess.graph)





    #writer = tf.summary.FileWriter("output/1")

    # for i in range(0,1):
    #     for j in range(0,10):
    #         print("next_epoch")
    #         x_value, _ = mnist.train.next_batch(batch_size)
    #         x_value = tf.reshape(x_value,[batch_size,img_height, img_width, img_layer])
    #         sess.run(g_trainer,feed_dict={x_data:x_value.eval(session=sess)})

    #writer.add_graph(sess.graph)












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

#     if to_restore:
#         chkpt_fname = tf.train.latest_checkpoint(output_path)
#         saver.restore(sess, chkpt_fname)
#     else:
#         if os.path.exists(output_path):
#             shutil.rmtree(output_path)
#         os.mkdir(output_path)


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