import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
from kdtree import make_cKDTree
from datasets import PartDataset

import sys
sys.path.append('/home/krabec/models/PNET2')
import modelnet_dataset
import modelnet_h5_dataset

num_points = 2048
BATCH_SIZE = 16

def print_tree(tree, cutdim):
    for i in range(len(cutdim)):
        print(len(tree[i]), len(cutdim[i]))
    print(len(tree), len(tree[11]))


class KDNet(nn.Module):
    def __init__(self, k=16):
        super(KDNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 8 * 3, 1, 1)
        self.conv2 = nn.Conv1d(8, 32 * 3, 1, 1)
        self.conv3 = nn.Conv1d(32, 64 * 3, 1, 1)
        self.conv4 = nn.Conv1d(64, 64 * 3, 1, 1)
        self.conv5 = nn.Conv1d(64, 64 * 3, 1, 1)
        self.conv6 = nn.Conv1d(64, 128 * 3, 1, 1)
        self.conv7 = nn.Conv1d(128, 256 * 3, 1, 1)
        self.conv8 = nn.Conv1d(256, 512 * 3, 1, 1)
        self.conv9 = nn.Conv1d(512, 512 * 3, 1, 1)
        self.conv10 = nn.Conv1d(512, 512 * 3, 1, 1)
        self.conv11 = nn.Conv1d(512, 1024 * 3, 1, 1)
        self.fc = nn.Linear(1024, 40)

    def forward(self, x, c):
        def kdconv(x, dim, featdim, sel, conv):
            batchsize = x.size(0)
            x = F.relu(conv(x))
            x = x.view(-1, featdim, 3, dim)
            x = x.view(-1, featdim, 3 * dim)
            sel = Variable(sel + (torch.arange(0, dim) * 3).long())
            if x.is_cuda:
                sel = sel.cuda()
            x = torch.index_select(x, dim=2, index=sel)
            x = x.view(-1, featdim, dim / 2, 2)
            x = torch.squeeze(torch.max(x, dim=-1, keepdim=True)[0], 3)
            return x

        x1 = kdconv(x, 2048, 8, c[-1], self.conv1)
        x2 = kdconv(x1, 1024, 32, c[-2], self.conv2)
        x3 = kdconv(x2, 512, 64, c[-3], self.conv3)
        x4 = kdconv(x3, 256, 64, c[-4], self.conv4)
        x5 = kdconv(x4, 128, 64, c[-5], self.conv5)
        x6 = kdconv(x5, 64, 128, c[-6], self.conv6)
        x7 = kdconv(x6, 32, 256, c[-7], self.conv7)
        x8 = kdconv(x7, 16, 512, c[-8], self.conv8)
        x9 = kdconv(x8, 8, 512, c[-9], self.conv9)
        x10 = kdconv(x9, 4, 512, c[-10], self.conv10)
        x11 = kdconv(x10, 2, 1024, c[-11], self.conv11)
        x11 = x11.view(-1, 1024)
        out = F.log_softmax(self.fc(x11))
        return out


def test(dataset, net):
    corrects = 0
    count = 0
    while dataset.has_next_batch():
        count+=1
        batch_data, batch_label = dataset.next_batch(augment=False)
        point_set, class_label = torch.from_numpy(np.reshape(batch_data,(num_points,3))), torch.from_numpy(batch_label.astype(np.long))
        point_set, class_label = torch.from_numpy(np.reshape(batch_data,(num_points,3))), torch.from_numpy(batch_label.astype(np.long))
        target = Variable(class_label).cuda()
        point_set = point_set[:num_points]
        cutdim, tree = make_cKDTree(point_set.numpy(), depth=levels)
        cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]
        points = torch.FloatTensor(tree[-1])
        points_v = Variable(torch.unsqueeze(torch.squeeze(points), 0)).transpose(2, 1).cuda()
        pred = net(points_v, cutdim_v)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        corrects += correct.item()
    print ('accuracy ' + str(float(corrects)/count*100))
    #if pred_choice.eq(target.data).cpu().sum() == 
#d = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0', classification=True)

BASE_DIR = '/home/krabec/models/PNET2'  
TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=num_points, shuffle=True)
TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=num_points, shuffle=False)
#l = len(d)
#print(len(d.classes), l)
levels = (np.log(num_points) / np.log(2)).astype(int)
net = KDNet().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for it in range(100000):
    optimizer.zero_grad()
    losses = []
    corrects = []
    count = 0
    for batch in range(32):
        if not TRAIN_DATASET.has_next_batch():
            print('reseting')
            TRAIN_DATASET.reset()
            TRAIN_DATASET.has_next_batch()
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
        count+=1
        #j = np.random.randint(l)
        point_set, class_label = torch.from_numpy(np.reshape(batch_data,(num_points,3))), torch.from_numpy(batch_label.astype(np.long))
        
        target = Variable(class_label).cuda()
        
        point_set = point_set[:num_points]
        if point_set.size(0) < num_points:
            print("too small")
            point_set = torch.cat([point_set, point_set[0:num_points - point_set.size(0)]], 0)

        cutdim, tree = make_cKDTree(point_set.numpy(), depth=levels)

        cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64))) for item in cutdim]

        #print_tree(tree, cutdim)

        points = torch.FloatTensor(tree[-1])
        points_v = Variable(torch.unsqueeze(torch.squeeze(points), 0)).transpose(2, 1).cuda()
        
        pred = net(points_v, cutdim_v)
        
        pred_choice = pred.data.max(1)[1]
        
        correct = pred_choice.eq(target.data).cpu().sum()
        
        corrects.append(correct)
        loss = F.nll_loss(pred, target)
        loss.backward()
        losses.append(loss.data[0])

    optimizer.step()
    #print('batch: %d, loss: %f, correct %d/%d' % (it, np.mean(losses), np.sum(corrects), count))
    if it % 1000 == 0:
        test(TEST_DATASET, net)
        TEST_DATASET.reset()
        TEST_DATASET.has_next_batch()
    if it % 5000 == 0:
        torch.save(net.state_dict(), 'save_model_%d.pth' % (it))
