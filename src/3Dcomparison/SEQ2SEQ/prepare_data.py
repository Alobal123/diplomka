import sys
import time
import numpy as np  
import tensorflow as tf
sys.path.append('/home/krabec/models/MVCNN/')
from input import Dataset
from model import _conv, _maxpool, _fc, _load_param

def alexnet(views, keep_prob):
    """
    views: N x V x W x H x C tensor
    """
    n_views = views.get_shape().as_list()[1] 

    # transpose views : (NxVxWxHxC) -> (VxNxWxHxC)
    views = tf.transpose(views, perm=[1, 0, 2, 3, 4])
    
    pool = [0]*n_views
    for i in xrange(n_views):
        # set reuse True for i > 0, for weight-sharing
        reuse = (i != 0)
        view = tf.gather(views, i) # NxWxHxC

        conv1 = _conv('conv1', view, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
        lrn1 = None
        pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = _conv('conv2', pool1, [5, 5, 96, 256], group=2, reuse=reuse)
        lrn2 = None
        pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
        conv4 = _conv('conv4', conv3, [3, 3, 384, 384], group=2, reuse=reuse)
        conv5 = _conv('conv5', conv4, [3, 3, 384, 256], group=2, reuse=reuse)

        pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        flat = tf.layers.flatten(pool5)

        fc6 = _fc('fc6', flat, 4096, dropout=keep_prob, reuse=reuse)
        fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob, reuse = reuse)
        print(fc7)
        print(pool)
        pool[i] = fc7
        
    return pool
    
def load_alexnet(sess, caffetf_modelpath):
    """ caffemodel: np.array, """
    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = l
        _load_param(sess, name, data_dict[l])

def extract_features(dataset, caffemodel):
    V = 20
    batch_size = 32

    data_size = dataset.size()
    print 'dataset size:', data_size

    features = np.zeros([data_size, 20, 4096])
    labels = np.zeros([data_size], dtype = np.int32)

    with tf.Graph().as_default():
        startstep = 0
        global_step = tf.Variable(startstep, trainable=False)

        view_ = tf.placeholder('float32', shape=(None, V, 227, 227, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        pool = alexnet(view_, keep_prob_)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        
        sess.run(init_op)
        load_alexnet(sess, caffemodel)

        step = startstep

        print "Start extracting"
        print "Size:", data_size

        for batch_x, batch_y in dataset.batches(batch_size):
            step += 1

            start_time = time.time()
            feed_dict = {view_: batch_x,
                         keep_prob_: 1.0}

            feature = sess.run(
                    [pool],
                    feed_dict=feed_dict)[0]

            feature = np.array(feature)
            feature = np.moveaxis(feature, 0, 1)
            if step%10 == 0:
                print(time.time() - start_time)
            
            offset = batch_y.shape[0]
            index = batch_size*(step-1)
            features[index:index+offset] = feature
            labels[index:index+offset] = batch_y
            
        return features, labels

def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels

def save_data(name, features, labels):
    np.save('./data/{}_features.npy'.format(name), features)
    np.save('./data/{}_labels.npy'.format(name), labels)

def create_data(listfiles, name):
    st = time.time()
    print 'start loading data'
    
    listfiles, labels = read_lists(listfiles)
    dataset = Dataset(listfiles, labels, subtract_mean=False, V=20)
    features, labels = extract_features(dataset, '/home/krabec/models/MVCNN/alexnet_imagenet.npy')
    save_data(name, features, labels)
    print 'done loading data, time=', time.time() - st
    

create_data('/home/krabec/models/MVCNN/datan/testfiles.txt', 'test')
create_data('/home/krabec/models/MVCNN/datan/trainfiles.txt', 'train')

