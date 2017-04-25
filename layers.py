import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="conv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[f_h, f_w, inputconv.get_shape()[-1], o_d], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(inputconv,filter=w,strides=[1,s_w,s_h,1],padding=padding)
        biases = tf.get_variable('b',[o_d],initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv,biases)
        if do_norm:
            conv = instance_norm(conv)
            # dims = conv.get_shape()
            # scale = tf.get_variable('scale',[dims[1],dims[2],dims[3]],initializer=tf.constant_initializer(1))
            # beta = tf.get_variable('beta',[dims[1],dims[2],dims[3]],initializer=tf.constant_initializer(0))
            # conv_mean,conv_var = tf.nn.moments(conv,[0])
            # conv = tf.nn.batch_normalization(conv,conv_mean,conv_var,beta,scale,0.001)
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

    return conv



def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding="VALID", name="deconv2d", do_norm=True, do_relu=True, relufactor=0):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[f_h, f_w, o_d, inputconv.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d_transpose(inputconv,filter=w,output_shape=outshape,strides=[1,s_w,s_h,1],padding=padding)
        biases = tf.get_variable('b',[o_d],initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv,biases)
        if do_norm:
            conv = instance_norm(conv)
            # dims = conv.get_shape()
            # scale = tf.get_variable('scale',[dims[1],dims[2],dims[3]],initializer=tf.constant_initializer(1))
            # beta = tf.get_variable('beta',[dims[1],dims[2],dims[3]],initializer=tf.constant_initializer(0))
            # conv_mean,conv_var = tf.nn.moments(conv,[0])
            # conv = tf.nn.batch_normalization(conv,conv_mean,conv_var,beta,scale,0.001)
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

    return conv
