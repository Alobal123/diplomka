from __future__ import print_function
import os
import numpy as np
import pandas as pd
import h5py as h5
import Shapenet
import re
from mesh_files import *

def get_file_id(file):
    return file.split(os.sep)[-3]


def  save_for_kdnet(files, config, categories, split):
    
    path2data = config.data
    path2save = config.output
    
    train_vertices_cnt = 0
    train_faces_cnt = 0
    test_vertices_cnt = 0
    test_faces_cnt = 0
    
    train_filenames = [ file for file in files if split[get_file_id(file)] == 0]
    test_filenames = [ file for file in files if split[get_file_id(file)] == 1]
    print('Train files : ' , len(train_filenames))
    print('Test files : ' , len(test_filenames))
    
    for i, shapefile in enumerate(test_filenames):
        if not i%100:
            print('Counting Test : {} out of {}'.format(i, len(test_filenames)))
        with open(shapefile, 'r',encoding='utf8') as fobj:
            for line in fobj:
                if 'v' in line:
                    test_vertices_cnt += 1
                if 'f' in line:
                    test_faces_cnt += 1 

    hf = h5.File(path2save + '/data.h5', 'w')

    test_nFaces = hf.create_dataset('test_nFaces', (1 + len(test_filenames),), dtype=np.int32)
    test_faces = hf.create_dataset('test_faces', (test_faces_cnt, 3), dtype=np.int32)
    test_vertices = hf.create_dataset('test_vertices', (test_vertices_cnt, 3), dtype=np.float32)
    test_labels = hf.create_dataset('test_labels', (len(test_filenames),), dtype=np.int8)
    test_nFaces[0] = 0
    vertices_pos = 0
    faces_pos = 0
    for i, shapefile in enumerate(test_filenames):
        if not i%100:
            print('Saving Test : {} out of {}'.format(i, len(test_filenames)))
        shape_vertices = []
        shape_faces = []
        with open(shapefile, 'r', encoding='utf8') as fobj:
            for j, line in enumerate(fobj):
                tmp = line.split(' ')
                if tmp[0] == 'v':
                    shape_vertices.append(list(map(np.float32, tmp[1:])))
                elif tmp[0] == 'f':
                    shape_faces.append(list(map(lambda x: np.int32(x.split(os.sep)[0]), tmp[2:])))

        shape_vertices = np.array(shape_vertices)
        buf = shape_vertices[:, 1].copy()
        shape_vertices[:, 1] = shape_vertices[:, 2]
        shape_vertices[:, 2] = buf
        shape_faces = np.array(shape_faces) - 1

        vertices_offset = shape_vertices.shape[0]
        faces_offset = shape_faces.shape[0]

        test_vertices[vertices_pos:vertices_pos+vertices_offset] = shape_vertices
        test_faces[faces_pos:faces_pos+faces_offset] = vertices_pos + shape_faces
        test_nFaces[i+1] = faces_pos + faces_offset
        test_labels[i] = categories[get_file_id(shapefile)]
        vertices_pos += vertices_offset
        faces_pos += faces_offset

    for i, shapefile in enumerate(train_filenames):
        if not i%100:
            print('Counting Train : {} out of {}'.format(i, len(train_filenames)))
        with open(shapefile, 'r',encoding='utf8') as fobj:
            for line in fobj:
                if 'v' in line:
                    train_vertices_cnt += 1
                if 'f' in line:
                    train_faces_cnt += 1
                    

    train_nFaces = hf.create_dataset('train_nFaces', (1 + len(train_filenames),), dtype=np.int32)
    train_faces = hf.create_dataset('train_faces', (train_faces_cnt, 3), dtype=np.int32)
    train_vertices = hf.create_dataset('train_vertices', (train_vertices_cnt, 3), dtype=np.float32)
    train_labels = hf.create_dataset('train_labels', (len(train_filenames),), dtype=np.int8)
    train_nFaces[0] = 0
    vertices_pos = 0
    faces_pos = 0
    for i, shapefile in enumerate(train_filenames):
        if not i%100:
            print('Saving Train : {} out of {}'.format(i, len(train_filenames)))
        shape_vertices = []
        shape_faces = []
        with open(shapefile, 'r', encoding='utf8') as fobj:
            for j, line in enumerate(fobj):
                tmp = line.strip().split(' ')
                if tmp[0] == 'v':
                    shape_vertices.append(list(map(np.float32, tmp[1:])))
                elif tmp[0] == 'f':
                    shape_faces.append(list(map(lambda x: np.int32(x.split(os.sep)[0]), tmp[2:])))

        shape_vertices = np.array(shape_vertices)
        buf = shape_vertices[:, 1].copy()
        shape_vertices[:, 1] = shape_vertices[:, 2]
        shape_vertices[:, 2] = buf
        shape_faces = np.array(shape_faces) - 1

        vertices_offset = shape_vertices.shape[0]
        faces_offset = shape_faces.shape[0]

        train_vertices[vertices_pos:vertices_pos+vertices_offset] = shape_vertices
        train_faces[faces_pos:faces_pos+faces_offset] = vertices_pos + shape_faces
        train_nFaces[i+1] = faces_pos + faces_offset
        train_labels[i] = categories[get_file_id(shapefile)]

        vertices_pos += vertices_offset
        faces_pos += faces_offset


    hf.close()
    print('\nData is processed and saved to ' + path2save + '/data.h5')


def prepare(config):
    path2data = config.data
    path2save = config.output
    categories, split, cat_names = Shapenet.get_metadata(path2data)
    files = find_files(path2data, 'obj')
    Shapenet.write_cat_names(path2data, path2save)
    save_for_kdnet(files,config, categories, split)


        