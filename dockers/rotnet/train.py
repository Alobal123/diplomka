#!/usr/bin/env python
from __future__ import print_function
import caffe
from my_classify_modelnet import classify
from my_classify_modelnet import get_mean
import numpy as np
import os
from Logger import Logger, log
from config import get_config, prepare_solver_file,  add_to_config
from prepare_data import prepare_data

def get_highest_model(config):
    files = os.listdir(config.log_dir)
    files = [os.path.join(config.log_dir,file) for file in files if os.path.splitext(file)[1] == '.solverstate']
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def get_dataset_size(config, name):
    file = os.path.join(config.data, '{}.txt'.format(name))
    with open(file, 'r') as f:
        count = len(f.readlines())
    return count

def eval(config, solver, epoch=0):
    acc = 0
    loss = 0
    labels = []
    predictions = []
    
    test_file = os.path.join(config.data, 'testrotnet.txt')
    mean = get_mean(config.mean_file)
    
    batch_size = solver.test_nets[0].blobs['data'].num
    test_iters = int(get_dataset_size(config, 'testrotnet') / config.batch_size)
    
    for i in range(test_iters):
        solver.test_nets[0].forward()
        acc += solver.test_nets[0].blobs['my_accuracy'].data
        loss += solver.test_nets[0].blobs['(automatic)'].data
        probs = solver.test_nets[0].blobs['prob'].data
        predictions+= classify(probs)
        labels.append(int(solver.test_nets[0].blobs['label'].data[0]))
        
    acc /= test_iters
    loss  /= test_iters
    if not config.test:
        log(config.log_file, "Accuracy: {:.3f}".format(acc))
        log(config.log_file, "Loss: {:.3f}".format(loss))
        LOSS_LOGGER.log( loss, epoch, "eval_loss")
        ACC_LOGGER.log( acc, epoch, "eval_accuracy")
        log(config.log_file, "EVALUATING epoch {} - loss: {} acc: {} ".format(epoch, loss, acc))
    else:
        import Evaluation_tools as et
        eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
        et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
        et.make_matrix(config.data, eval_file, config.log_dir) 
        

def train(config, solver):
    
    if config.weights == -1:
        startepoch = 0
        solver.net.copy_from('./caffe_nets/ilsvrc13')
    else:
        weights = config.weights
        startepoch = weights + 1
        ld = config.log_dir
        snapshot = os.path.join(config.snapshot_prefix[1:-1]+'_iter_'+str(weights))  
        ACC_LOGGER.load((os.path.join(ld,"{}_acc_train_accuracy.csv".format(config.name)),
                            os.path.join(ld,"{}_acc_eval_accuracy.csv".format(config.name))), epoch = weights)
        LOSS_LOGGER.load((os.path.join(ld,"{}_loss_train_loss.csv".format(config.name)),
                               os.path.join(ld,'{}_loss_eval_loss.csv'.format(config.name))), epoch = weights)
        solver.restore(snapshot + '.solverstate')
        solver.net.copy_from(snapshot + '.caffemodel')
        solver.test_nets[0].copy_from(snapshot + '.caffemodel')
    
    steps_per_epoch = get_dataset_size(config, 'trainrotnet') / solver.net.blobs['data'].data.shape[0] + 1
    for epoch in range(startepoch, startepoch + config.max_iter + 1):
        eval(config, solver, epoch=epoch)            
        losses = []
        accs = []
        for it in range(steps_per_epoch):
            
            solver.step(1)
            loss = float(solver.net.blobs['(automatic)'].data)
            acc = float(solver.net.blobs['my_accuracy'].data)
            losses.append(loss)
            accs.append(acc)
            
            if it % max(config.train_log_frq/config.batch_size,1) == 0:
                loss = np.mean(losses)
                acc = np.mean(accs)
                LOSS_LOGGER.log( loss, epoch, "train_loss")
                ACC_LOGGER.log( acc , epoch, "train_accuracy")
                ACC_LOGGER.save(config.log_dir)
                LOSS_LOGGER.save(config.log_dir)
                losses = []
                accs = []
                highest_model_saved = it
                log(config.log_file, "TRAINING epoch: {} it: {}  loss: {} acc: {} ".format(epoch,it, loss, acc))
        ACC_LOGGER.plot(dest=config.log_dir)
        LOSS_LOGGER.plot(dest=config.log_dir)        


def set_num_cats(config):
    import prepare_nets
    prepare_nets.set_num_cats(config.train_net[1:-1], config.num_classes, config.num_views)
    prepare_nets.set_num_cats(config.test_net[1:-1],config.num_classes, config.num_views)
    prepare_nets.set_batch_size(config.train_net[1:-1], config.batch_size)
    
if __name__ == '__main__':
    
    config = get_config()
    set_num_cats(config)
    
    with open(config.log_file, 'w') as f:
        print('Starting')
        print('Starting', file=f)
    
    prepare_data(os.path.join(config.data, 'train.txt'), views=config.num_views, shuffle=True)
    prepare_data(os.path.join(config.data, 'test.txt'), views=config.num_views, shuffle=False)
    
    caffe.set_device(0)
    caffe.set_mode_gpu()
    data_size = get_dataset_size(config, 'trainrotnet')
    prepare_solver_file(data_size=data_size)
    solver = caffe.get_solver(config.solver)
    
    if not config.test:
        LOSS_LOGGER = Logger("{}_loss".format(config.name))
        ACC_LOGGER = Logger("{}_acc".format(config.name))
        train(config, solver)
        config = add_to_config(config, 'weights', highest_model_saved)
        config = add_to_config(config, 'test', True)
        
    snapshot = get_highest_model(config)

    solver.restore(snapshot)
    caffemodel = os.path.splitext(snapshot)[0] + '.caffemodel'
    solver.net.copy_from(caffemodel)
    solver.test_nets[0].copy_from(caffemodel)
    log(config.log_file, 'Model restored')
    eval(config, solver)
    