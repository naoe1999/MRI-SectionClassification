from data_loader import read_data, Dataset
from skimage import io
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
import cv2
import os


def vgg_arg_scope(weight_decay=0.0005, batch_norm=False, is_training=False):
    batch_norm_params = {
        'is_training': is_training,
        'trainable': is_training
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        normalizer_fn=slim.batch_norm if batch_norm else None,
                        normalizer_params=batch_norm_params if batch_norm else None):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc


def vgg_16(x, is_training=True, scope='vgg_16'):
    """Oxford Net VGG 16-Layers version D Example.
    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
    Returns:
    net: feature map of conv. layer.
    end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [x]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], trainable=is_training, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=is_training, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


def gap_net(x, nclasses, is_training=True):
    feature = slim.conv2d(x, 512, [3, 3], trainable=is_training, scope='conv6')
    # net = slim.dropout(net, keep_prob=0.5, is_training=is_training)

    net = slim.avg_pool2d(feature, [AVGPOOLSIZE, AVGPOOLSIZE],
                          stride=[AVGPOOLSIZE, AVGPOOLSIZE], padding='SAME', scope='gap')
    # net = slim.dropout(net, keep_prob=0.5, is_training=is_training)

    net = tf.reshape(net, [-1, 512])

    net = slim.fully_connected(net, nclasses, trainable=is_training, scope='fc')
    return feature, net


def cam(x):
    # x: shape: (n, 14, 14, 512)
    weight = slim.get_variables('fc/weights:0')         # shape: (512, ncls)
    weight = tf.reshape(weight, (1, 1, 1, 512, -1))     # shape: (1, 1, 1, 512, ncls)

    x = tf.expand_dims(x, axis=-1)                      # shape: (n, 14, 14, 512, 1)

    cmap = x * weight                                   # shape: (n, 14, 14, 512, ncls)
    cmap = tf.reduce_mean(cmap, axis=3)                 # shape: (n, 14, 14, ncls)
    return cmap


if __name__ == '__main__':

    # hyper-parameters
    NUMCLASSES = 5
    GRAYSCALE = False
    IMAGESIZE = (224, 224)
    AVGPOOLSIZE = 14

    TRAINING = True
    USEPRETRAIN = True
    USEBATCHNORM = False
    VALIDATIONRATE = 0.1
    LR = 0.000005
    EPOCH = 50
    NUMDATA = 0
    BATCH = 20
    LOGSTEP = 10

    DATASET = '5classification'
    EXECNAME = 'take01_50ep'
    IMGEXT = 'jpg'
    TRAINDATA = os.path.join('.\\images\\train', DATASET)
    TESTDATA = os.path.join('.\\images\\test', DATASET)
    LOGDIR = os.path.join('D:\\tensorboard\\vet', DATASET)
    SAVEDIR = os.path.join('.\\results', DATASET, 'model')
    CAMDIR = os.path.join('.\\results', DATASET, 'cam')

    if not os.path.isdir(SAVEDIR):
        os.makedirs(SAVEDIR)
    if not os.path.isdir(CAMDIR):
        os.makedirs(CAMDIR)

    ##########################
    # build model
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, None, None, 3], name='x')
        y = tf.placeholder(tf.int32, [None], name='y')

    # CNN
    with slim.arg_scope(vgg_arg_scope(batch_norm=USEBATCHNORM, is_training=TRAINING)):
        conv_feat, _ = vgg_16(x, is_training=TRAINING)

    if TRAINING and USEPRETRAIN:
        init_fn = slim.assign_from_checkpoint_fn(
            model_path='vgg_16.ckpt', var_list=slim.get_model_variables('vgg_16'), ignore_missing_vars=True
        )
    else:
        init_fn = None

    # classification
    last_feat, prediction = gap_net(conv_feat, NUMCLASSES, is_training=TRAINING)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    # optimizer
    if TRAINING:
        global_step = tf.train.create_global_step()
        optm = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost, global_step=global_step)
        # optm = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(cost, global_step=global_step)
    else:
        global_step, optm = None, None

    # metric: accuracy
    pred = tf.cast(tf.argmax(prediction, 1), tf.int32)
    correct = tf.equal(pred, y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # class activation map
    cmap = cam(last_feat)

    ##########################
    # define summary
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)
    summaries = tf.summary.merge_all()

    saver = tf.train.Saver()

    ##########################
    # load data
    if TRAINING:
        imgs, labels, ncls, _ = read_data(
            TRAINDATA, limit=NUMDATA, img_ext=IMGEXT, size=IMAGESIZE, grayscale=GRAYSCALE
        )
        assert ncls == NUMCLASSES

        # split data
        ndata = len(imgs)
        ntrain = int(ndata * (1.0 - VALIDATIONRATE))

        randidx = np.random.permutation(ndata)
        trainidx = randidx[:ntrain]
        valididx = randidx[ntrain:]

        train_imgs = [imgs[i] for i in trainidx]
        train_labels = [labels[i] for i in trainidx]

        valid_imgs = [imgs[i] for i in valididx]
        valid_labels = [labels[i] for i in valididx]

        train_ds = Dataset(train_imgs, train_labels)
        valid_ds = Dataset(valid_imgs, valid_labels)

    else:
        train_ds = None
        valid_ds = None

    ##########################
    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    ##########################
    # train
    if TRAINING:
        if init_fn is not None:
            init_fn(sess)
            print('pre-trained weights have been loaded.')

        log_path = os.path.join(LOGDIR, EXECNAME)
        save_path = os.path.join(SAVEDIR, EXECNAME, EXECNAME)

        train_writer = tf.summary.FileWriter(os.path.join(log_path, 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(log_path, 'valid'))

        for epoch in range(1, EPOCH + 1):

            while train_ds.cur_epoch <= epoch:
                step = sess.run(global_step)
                imgs, labels = train_ds.next_batch(BATCH, shuffle=True)

                if step % LOGSTEP == 0:
                    # training
                    _, loss, acc, summary = sess.run([optm, cost, accuracy, summaries],
                                                     feed_dict={x: imgs, y: labels})

                    train_writer.add_summary(summary, step)
                    train_writer.flush()
                    print('TRAIN: epoch: {}, loss: {}, accuracy: {}'.format(epoch, loss, acc))

                    # validation
                    if not valid_ds.is_empty:
                        imgs, labels = valid_ds.next_batch(BATCH, shuffle=True)
                        loss, acc, summary = sess.run([cost, accuracy, summaries], feed_dict={x: imgs, y: labels})

                        valid_writer.add_summary(summary, step)
                        valid_writer.flush()
                        print('VALID: epoch: {}, loss: {}, accuracy: {}'.format(epoch, loss, acc))
                        print('')

                else:
                    sess.run(optm, feed_dict={x: imgs, y: labels})

            # save checkpoint
            step = sess.run(global_step)
            saver.save(sess, save_path, global_step=step, write_meta_graph=True)
            print('model saved.')

        train_writer.close()
        valid_writer.close()

    else:
        # restore checkpoint
        save_path = os.path.join(SAVEDIR, EXECNAME)
        saver.restore(sess, tf.train.latest_checkpoint(save_path))

    ##########################
    # evaluate
    imgs, labels, _, cls_names = read_data(TESTDATA, img_ext=IMGEXT, size=IMAGESIZE, grayscale=GRAYSCALE)
    test_ds = Dataset(imgs, labels)

    succ_cam_path = os.path.join(CAMDIR, EXECNAME, 'succeeded')
    if not os.path.exists(succ_cam_path):
        os.makedirs(succ_cam_path)

    fail_cam_path = os.path.join(CAMDIR, EXECNAME, 'failed')
    if not os.path.exists(fail_cam_path):
        os.makedirs(fail_cam_path)

    eval_batch_size = 10

    avg_acc, avg_loss = 0.0, 0.0
    iteration = 0

    while True:
        imgs, labels = test_ds.next_batch(eval_batch_size, shuffle=False)

        if test_ds.cur_epoch > 1:
            break

        loss, acc, preds, corrs, cmaps = sess.run([cost, accuracy, pred, correct, cmap],
                                                  feed_dict={x: imgs, y: labels})

        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :]
            t = labels[i]
            p = preds[i]
            c = corrs[i]
            assert c == (t == p)

            heatmap = cmaps[i, :, :, p]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap = cv2.resize(heatmap, IMAGESIZE)
            heatmap = np.uint8(heatmap * 255)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)

            inputimg = np.uint8(img * 255)
            blended = np.uint8(inputimg * 0.7 + heatmap * 0.3)

            true_class = cls_names[t]
            pred_class = cls_names[p]

            seq_num = eval_batch_size * iteration + i + 1
            filename = 'result_{}_t{}_p{}.jpg'.format(seq_num, true_class, pred_class)
            filepath = os.path.join(succ_cam_path if c else fail_cam_path, filename)

            io.imsave(filepath, blended)

        avg_acc += acc
        avg_loss += loss
        iteration += 1

    avg_acc /= iteration
    avg_loss /= iteration

    print('Iteration: {}, test accuracy: {}, test loss: {}'.format(iteration, avg_acc, avg_loss))

