# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil


img_height = 28
img_width = 28
img_layer = 1
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


# def build_generator(z_prior):
#     w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
#     b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
#     h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)
#     w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
#     b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
#     h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
#     w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
#     b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)
#     h3 = tf.matmul(h2, w3) + b3
#     x_generate = tf.nn.tanh(h3)
#     g_params = [w1, b1, w2, b2, w3, b3]
#     return x_generate, g_params


# def build_discriminator(x_data, x_generated, keep_prob):
#     x_in = tf.concat(0, [x_data, x_generated])
#     w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
#     b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)
#     h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
#     w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
#     b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
#     h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
#     w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
#     b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
#     h3 = tf.matmul(h2, w3) + b3
#     y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
#     y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
#     d_params = [w1, b1, w2, b2, w3, b3]
#     return y_data, y_generated, d_params


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding=None, name="conv2d", do_norm=True, do_relu=True):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[f_h, f_w, inputconv.get_shape()[-1], o_d], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(inputconv,filter=w,strides=[1,s_w,s_h,1],padding=padding)
        biases = tf.get_variable('b',[o_d],initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv,biases)
        if do_norm:
            dims = conv.get_shape()
            scale = tf.get_variable('scale',[dims[1],dims[2],dims[3]],initializer=tf.constant_initializer(1))
            beta = tf.get_variable('beta',[dims[1],dims[2],dims[3]],initializer=tf.constant_initializer(0))
            conv_mean,conv_var = tf.nn.moments(conv,[0])
            conv = tf.nn.batch_normalization(conv,conv_mean,conv_var,beta,scale,0.001)
        if do_relu:
            conv = tf.nn.relu(conv, relufactor, "relu")

    return conv

def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding=None, name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[f_h, f_w, o_d, inputconv.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d_transpose(inputconv,filter=w,output_shape=outshape,strides=[1,s_w,s_h,1],padding=padding)
        biases = tf.get_variable('b',[o_d],initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv,biases)
        if do_norm:
            dims = conv.get_shape()
            scale = tf.get_variable('scale',[dims[1],dims[2],dims[3]],initializer=tf.constant_initializer(1))
            beta = tf.get_variable('beta',[dims[1],dims[2],dims[3]],initializer=tf.constant_initializer(0))
            conv_mean,conv_var = tf.nn.moments(conv,[0])
            conv = tf.nn.batch_normalization(conv,conv_mean,conv_var,beta,scale,0.001)
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv,"lrelu")

    return conv

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


    # Load Dataset from the dataset folder

    filenames = tf.train.match_filenames_once("./A/*.jpg")
    queue_length = tf.size(filenames)
    filename_queue = tf.train.string_input_producer(filenames)
    image_reader = tf.WholeFileReader()

    _, image_file = image_reader.read(filename_queue)

    image = tf.image.decode_jpeg(image_file)

    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # x_data = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, img_layer), name="x_data")
    # x_generated = build_generator_resnet_6blocks(x_data,"g_1")
    # g_loss = tf.reduce_sum(x_generated)
    # g_trainer = tf.train.AdamOptimizer(0.001).minimize(g_loss,var_list=tf.trainable_variables())

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_files = sess.run(queue_length)

        images = []

        for i in range(num_files):
            image_tensor = sess.run(image)
            images.append(image_tensor)

        image_batch = tf.stack(images)

        num_images = image_batch.shape

        print(num_images)

        coord.request_stop()
        coord.join(threads)


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