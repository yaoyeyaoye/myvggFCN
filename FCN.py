# -*- coding:utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
# import read_MITSceneParsingData as scene_parsing
import myDataReader as dataset

import datetime
# import xrange

FLAGS = tf.flags.FLAGS #flags是一个文件：flags.py，用于处理命令行参数的解析工作
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "myvggFCN/Data_zoo/FlowData/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool("debug", "False", "Debug mode: True/ False")
tf.flags.DEFINE_string("mode", "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_string("zone", "SYMMETRY-5", "Mode train/ test/ visualize")

##############  general variable  ##############

TRAINING_PATH = '/Users/yaoye/PycharmProjects/myvggFCN/Data_zoo/FlowData/training'
VALIDATION_PATH = '/Users/yaoye/PycharmProjects/myvggFCN/Data_zoo/FlowData/validation'


MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3','relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3','relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3','relu5_3', 'conv5_4', 'relu5_4'
    )
    '''
    weights[i][0][0][0][0]:

    <tf.Variable 'inference/conv1_1_w:0' shape=(3, 3, 3, 64) dtype=float32_ref>
    <tf.Variable 'inference/conv1_1_b:0' shape=(64,) dtype=float32_ref>
    <tf.Variable 'inference/conv1_2_w:0' shape=(3, 3, 64, 64) dtype=float32_ref>
    <tf.Variable 'inference/conv1_2_b:0' shape=(64,) dtype=float32_ref>

    <tf.Variable 'inference/conv2_1_w:0' shape=(3, 3, 64, 128) dtype=float32_ref>
    <tf.Variable 'inference/conv2_1_b:0' shape=(128,) dtype=float32_ref>
    <tf.Variable 'inference/conv2_2_w:0' shape=(3, 3, 128, 128) dtype=float32_ref>
    <tf.Variable 'inference/conv2_2_b:0' shape=(128,) dtype=float32_ref>

    '''

    net = {}
    current = image
    for i, name in enumerate(layers): # 对于一个可迭代/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            print("kernels:",i,kernels)
            # print kernels
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            print(kernels)
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            print(bias)
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL) # model_data 是dict型
    '''
    model_data type: <type 'dict'>
    key: layers         value : <type 'numpy.ndarray'>
    key: __header__     value : <type 'str'>
    key: __globals__    value : <type 'list'>
    key: classes        value : <type 'numpy.ndarray'>
    key: __version__    value :  "1.0"   <type 'str'>
    key: normalization  value : <type 'numpy.ndarray'>
    '''
    # print(model_data['classes'])

    mean = model_data['normalization'][0][0][0] # (224, 224, 3)
    mean_pixel = np.mean(mean , axis=(0, 1))
    # print (model_data['normalization'])

    weights = np.squeeze(model_data['layers']) # np.squeeze：从数组的形状中删除单维条目，即把shape中为1的维度去掉

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        # 定义前5层的vggnet
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        pool5 = utils.max_pool_2x2(conv_final_layer)
        # 从第六层开始往下定义
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6") # [patch 7*7,insize 512, outsize 4096]
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6) # outsize
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss_val, var_list=var_list) # 该函数为函数minimize()的第一部分：对var_list中的变量计算loss的梯度，返回一个以元组(gradient, variable)组成的列表
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads_and_vars:
            utils.add_gradient_summary(grad, var) # 是函数minimize()的第二部分：将计算出的梯度应用到变量上，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
    return optimizer.apply_gradients(grads_and_vars)


def main(argv=None):
    # define placeholder for inputs layer #
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    # create model structure #
    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    # Define loss and optimizer #
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy"))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables() # 返回需要训练的变量列表
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    # make up some real data #
    # print("Setting up image reader...")
    # train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir) # FLAGS.data_dir = Data_zoo/MIT_SceneParsing/


    print("Setting up dataset reader")
    image_options = {'image_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(TRAINING_PATH,FLAGS.zone, image_options)
    validation_dataset_reader = dataset.BatchDatset(VALIDATION_PATH,FLAGS.zone, image_options)


    # start to run the graph #
    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run() # 处理flag解析，然后执行main函数
