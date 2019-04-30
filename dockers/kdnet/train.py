from __future__ import print_function

import h5py as h5
import numpy as np
import os
from kdnet import KDNET
from Logger import Logger, log
from lib.generators.meshgrid import generate_clouds
from lib.trees.kdtrees import KDTrees
from lib.nn.utils import dump_weights, load_weights
import config as cfg

def iterate_minibatches(config, *arrays):
    if config['mode'] == 'train':
        indices = np.random.choice((len(arrays[2]) - 1), 
                                   size=(len(arrays[2]) - 1)/config['batch_size']*config['batch_size'])
    elif config['mode'] == 'test':
        indices = np.arange(len(arrays[2]) - 1)
    #indices = np.random.choice((len(arrays[2]) - 1), size=(len(arrays[2]) - 1)/config['batch_size']*config['batch_size'])
    if config['mode']=='train' and config['shuffle']:
        np.random.shuffle(indices)
        
    for start_idx in xrange(0, len(indices), config['batch_size']):
        excerpt = indices[start_idx:start_idx + config['batch_size']]
        tmp = generate_clouds(excerpt, config['steps'], arrays[0], arrays[1], arrays[2])
        
        if config['flip']:
            flip = np.random.random(size=(len(tmp), 2, 1))
            flip[flip >= 0.5] = 1.
            flip[flip < 0.5] = -1.
            tmp[:, :2] *= flip
        
        if config['ascale']:
            tmp *= (config['as_min'] + (config['as_max'] - config['as_min'])*np.random.random(size=(len(tmp), config['dim'], 1)))
            tmp /= np.fabs(tmp).max(axis=(1, 2), keepdims=True)
        if config['rotate']:
            r = np.sqrt((tmp[:, :2]**2).sum(axis=1))
            coss = tmp[:, 0]/r
            sins = tmp[:, 1]/r
            
            if config['test_pos'] is not None:
                alpha = 2*np.pi*config['test_pos']/config['r_positions']
            else:
                alpha = 2*np.pi*np.random.randint(0, config['r_positions'], (len(tmp), 1))/config['positions']
                
            cosr = np.cos(alpha)
            sinr = np.sin(alpha)
            cos = coss*cosr - sins*sinr
            sin = sins*cosr + sinr*coss
            tmp[:, 0] = r*cos
            tmp[:, 1] = r*sin
            
        if config['translate']:
            mins = tmp.min(axis=2, keepdims=True)
            maxs = tmp.max(axis=2, keepdims=True)
            rngs = maxs - mins
            tmp += config['t_rate']*(np.random.random(size=(len(tmp), config['dim'], 1)) - 0.5)*rngs
        
        trees_data = KDTrees(tmp, dim=config['dim'], steps=config['steps'], 
                             lim=config['lim'], det=config['det'], gamma=config['gamma'])
            
        sortings, normals = trees_data['sortings'], trees_data['normals']
        if config['input_features'] == 'all':
            clouds = np.empty((len(excerpt), config['dim'], 2**config['steps']), dtype=np.float32)
            for i, srt in enumerate(sortings):
                clouds[i] = tmp[i, :, srt].T
        elif config['input_features'] == 'no':
            clouds = np.ones((len(excerpt), 1, 2**config['steps']), dtype=np.float32)
        
        if config['mode'] == 'train':
            yield [clouds] + normals[::-1] + [arrays[3][excerpt]]
        if config['mode'] == 'test':
            yield [clouds] + normals[::-1] + [arrays[3][excerpt]]


def get_probs(net, vertices, faces, nFaces, labels, config):
    prob_sum = np.zeros((len(nFaces)-1, config['num_classes']), dtype=np.float32)
    losses = []
    for ens in xrange(config['num_votes']):
        probability = np.zeros((len(nFaces)-1, config['num_classes']), dtype=np.float32)
        index = 0    
        for i, batch in enumerate(iterate_minibatches(config, vertices, faces, nFaces,labels)):
            loss, probs = net.prob_fun(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11])
            losses.append(loss)
            size_of_batch = batch[-1].shape[0]
            probability[index:index+size_of_batch] += probs
            index += size_of_batch
            #probability[batch[-1]] += probs
        prob_sum += probability

    return np.mean(losses), prob_sum / config['num_votes']


def acc_fun(net, vertices, faces, nFaces, labels, config):
    loss, probs= get_probs(net, vertices, faces, nFaces, labels, config)
    return loss, probs.argmax(axis=1)

def test(config,test_vertices, test_faces, test_nFaces, test_labels):
    log(config['log_file'],"Start testing")
    config['mode'] = 'test'
    _, predictions = acc_fun(net,test_vertices, test_faces, test_nFaces, test_labels, config) 
    acc = 100.*(predictions == test_labels).sum()/len(test_labels)
    
    log(config['log_file'],'Eval accuracy:  {}'.format(acc))
    import Evaluation_tools as et
    eval_file = os.path.join(config['log_dir'], '{}.txt'.format(config['name']))
    print(eval_file)
    et.write_eval_file(config['data'], eval_file, predictions , test_labels , config['name'])
    et.make_matrix(config['data'], eval_file, config['log_dir'])

if __name__ == "__main__":
    config = cfg.config('config.ini').dictionary
    config['n_f'] =  [16, 32,  32,  64,  64,  128, 128, 256, 256, 512, 128]
    log(config['log_file'],"Reading data...")
    path2data = os.path.join(config['data'], 'data.h5')
    with h5.File(path2data, 'r') as hf:
        if not config['test']:
            train_vertices = np.array(hf.get('train_vertices'))
            train_faces = np.array(hf.get('train_faces'))
            train_nFaces = np.array(hf.get('train_nFaces'))
            train_labels = np.array(hf.get('train_labels'))
        test_vertices = np.array(hf.get('test_vertices'))
        test_faces = np.array(hf.get('test_faces'))
        test_nFaces = np.array(hf.get('test_nFaces'))
        test_labels = np.array(hf.get('test_labels'))
            
    log(config['log_file'], "Compiling net...")
    net = KDNET(config)  
    
    if config['weights']!=-1:
        weights = config['weights']
        load_weights(os.path.join(config['log_dir'], config['snapshot_prefix']+str(weights)), net.KDNet['output'])
        log(config['log_file'],"Loaded weights")
    
    if config['test']:
        test(config,test_vertices, test_faces, test_nFaces, test_labels)
        
    else:
        log(config['log_file'], "Start training")
        LOSS_LOGGER = Logger("kdnet_loss")
        ACC_LOGGER = Logger("kdnet_acc")
        start_epoch = 0
        if config['weights']!=-1:
            start_epoch = weights
            ACC_LOGGER.load((os.path.join(config['log_dir'],"kdnet_acc_train_accuracy.csv"),os.path.join(config['log_dir'],"kdnet_acc_eval_accuracy.csv")), epoch=weights)
            LOSS_LOGGER.load((os.path.join(config['log_dir'],"kdnet_loss_train_loss.csv"), os.path.join(config['log_dir'],'kdnet_loss_eval_loss.csv')), epoch=weights)
        
        num_save = config['save_period']
        begin = start_epoch
        end = config['max_epoch'] + start_epoch
        for epoch in xrange(begin, end + 1):
            
            config['mode'] = 'test'
            loss, predictions = acc_fun(net,test_vertices, test_faces, test_nFaces, test_labels, config)
            acc = (predictions == test_labels).sum()/float(len(test_labels))
            log(config['log_file'],'evaluating loss:{} acc:{}'.format(loss,acc))       
            LOSS_LOGGER.log(loss, epoch, "eval_loss")
            ACC_LOGGER.log(acc, epoch, "eval_accuracy")
            
            config['mode'] = 'train'
            losses = []
            accuracies = []
            for i, batch in enumerate(iterate_minibatches(config, train_vertices, train_faces, train_nFaces, train_labels)):
                train_err_batch, train_acc_batch = net.train_fun(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11])
                
                losses.append(train_err_batch)
                accuracies.append(train_acc_batch)

                if i % max(config['train_log_frq']/config['batch_size'], 1) == 0:
                    loss = np.mean(losses)
                    acc = np.mean(accuracies)
                    LOSS_LOGGER.log(loss, epoch, "train_loss")
                    ACC_LOGGER.log(acc, epoch, "train_accuracy")
                    log(config['log_file'],'EPOCH {}, batch {}: loss {} acc {}'.format(epoch, i, loss, acc))
                    losses = []
                    accuracies = []
            
            ACC_LOGGER.save(config['log_dir'])
            LOSS_LOGGER.save(config['log_dir'])
            ACC_LOGGER.plot(dest=config['log_dir'])
            LOSS_LOGGER.plot(dest=config['log_dir'])
            if epoch % num_save == 0 or epoch==end:
                dump_weights(os.path.join(config['log_dir'], config['snapshot_prefix']+str(epoch)), net.KDNet['output'])
                
        config['mode'] = 'test'      
        test(config,test_vertices, test_faces, test_nFaces, test_labels)
                