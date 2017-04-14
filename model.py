# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

from layers import *

img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = "output"



max_epoch = 500

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256
ngf = 128


def build_generator(z_prior):
    w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    x_generate = tf.nn.tanh(h3)
    g_params = [w1, b1, w2, b2, w3, b3]
    return x_generate, g_params


def build_discriminator(x_data, x_generated, keep_prob):
    x_in = tf.concat(0, [x_data, x_generated])
    w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
    d_params = [w1, b1, w2, b2, w3, b3]
    return y_data, y_generated, d_params

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

        o_c4 = general_deconv2d(o_r6, [batch_size,14,14,ngf*2], ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
        o_c5 = general_deconv2d(o_c4, [batch_size,28,28,ngf], ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
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
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    x_generated, g_params = build_generator(z_prior)
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)

    optimizer = tf.train.AdamOptimizer(0.0001)

    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)


    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

    for i in range(sess.run(global_step), max_epoch):
        for j in range(60000 / batch_size):
            print "epoch:%s, iter:%s" % (i, j)
            x_value, _ = mnist.train.next_batch(batch_size)
            x_value = 2 * x_value.astype(np.float32) - 1
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            sess.run(d_trainer,
                     feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            if j % 1 == 0:
                sess.run(g_trainer,
                         feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        show_result(x_gen_val, "output/sample{0}.jpg".format(i))
        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


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