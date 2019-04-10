#!/usr/bin/env python
from __future__ import print_function
import caffe
import numpy as np
import os
import lmdb
from Logger import Logger, log
from config import get_config, prepare_solver_file, add_to_config

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
    all_labels = []
    predictions = []
    test_count = get_dataset_size(config, 'test')
    keys = solver.test_nets[0].blobs.keys()
    batch_size = (solver.test_nets[0].blobs['label_octreedatabase_1_split_0'].data.shape[0]) 
    test_iters = test_count / batch_size
    
    logits = np.zeros((test_count, config.num_classes))
    print(logits.shape)
    for i in range(test_iters):
        solver.test_nets[0].forward()
        loss += solver.test_nets[0].blobs['loss'].data
        probs = solver.test_nets[0].blobs['ip2'].data
        
        logits[i*batch_size:(i+1)*batch_size] = probs
        all_labels += list(solver.test_nets[0].blobs['label'].data) 

    solver.test_nets[0].forward()
    loss += solver.test_nets[0].blobs['loss'].data
    probs = solver.test_nets[0].blobs['ip2'].data
    logits[batch_size*(test_iters):] = probs[0 : test_count % batch_size]
    all_labels += list(solver.test_nets[0].blobs['label'].data)[0 : test_count % batch_size]
       
    loss  /= test_iters + 1
    
    predictions = []
    labels = []
    
    for i in range(len(all_labels) / config.num_rotations):
        predictions.append( np.argmax( np.sum(logits[i*config.num_rotations : (i+1)*config.num_rotations], axis = 0 ) ) )
        labels.append(all_labels[i*config.num_rotations])
    
    acc = sum([1  for i in range(len(labels)) if predictions[i] == labels[i]])/float(len(labels))
    
    if not config.test:
        log(config.log_file, "EPOCH: {} Test loss: {}".format(epoch, loss))
        log(config.log_file, "EPOCH: {} Test accuracy: {}".format(epoch, acc))
        LOSS_LOGGER.log( loss, epoch, "eval_loss")
        ACC_LOGGER.log( acc, epoch, "eval_accuracy")
    else:
        log(config.log_file, "----------------------") 
        import Evaluation_tools as et
        labels = [int(l) for l in labels]
        eval_file = os.path.join(config.log_dir, '{}.txt'.format(config.name))
        et.write_eval_file(config.data, eval_file, predictions, labels, config.name)
        et.make_matrix(config.data, eval_file, config.log_dir) 

def set_num_cats(config):
    import prepare_nets
    prepare_nets.set_num_cats(config.net[1:-1], config.num_classes, 0)
    prepare_nets.set_batch_size(config.net[1:-1], config.batch_size)

def train(config, solver):
    
    if config.weights == -1:
        startepoch = 0
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
        
    steps_per_epoch = get_dataset_size(config, 'train') / config.batch_size

    begin = startepoch
    end = startepoch + config.max_iter + 1
    for epoch in range(begin, end):
        eval(config, solver, epoch=epoch)            
        losses = []
        accs = []
        for it in range(steps_per_epoch):
            solver.step(1)
            loss = float(solver.net.blobs['loss'].data)
            acc = float(solver.net.blobs['accuracy'].data)
            losses.append(loss)
            accs.append(acc)
            
            if it % max(config.train_log_frq/config.batch_size,1) == 0:
                LOSS_LOGGER.log( np.mean(losses), epoch, "train_loss")
                ACC_LOGGER.log( np.mean(accs), epoch, "train_accuracy")
                ACC_LOGGER.save(config.log_dir)
                LOSS_LOGGER.save(config.log_dir)
                losses = []
                accs = []
                highest_model_saved = it
                
        ACC_LOGGER.plot(dest=config.log_dir)
        LOSS_LOGGER.plot(dest=config.log_dir)        
        log(config.log_file, "LOSS: {}".format( np.mean(losses)))
        log(config.log_file, "ACCURACY {}".format(np.mean(accs)))


if __name__ == '__main__':
    config = get_config()
    set_num_cats(config)
    with open(config.log_file, 'w') as f:
        print("STARTING", file=f)
        print("STARTING")
    data_size = get_dataset_size(config, 'train')
    prepare_solver_file(data_size=data_size)
    set_num_cats(config)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.get_solver(config.solver)

    if not config.test:
        LOSS_LOGGER = Logger("{}_loss".format(config.name))
        ACC_LOGGER = Logger("{}_acc".format(config.name))
        train(config, solver)
        snapshot = get_highest_model(config)
        solver.restore(snapshot)
        config = add_to_config(config, 'test', True)
        eval(config, solver)
    else:
        weights = config.weights
        solver.restore(snapshot)
        snapshot = os.path.join(config.snapshot_prefix[1:-1]+'_iter_'+str(weights))
        solver.restore(snapshot + '.solverstate')
        solver.test_nets[0].copy_from(snapshot + '.caffemodel')
        log(config.log_file, 'Model restored')
        eval(config, solver)
        