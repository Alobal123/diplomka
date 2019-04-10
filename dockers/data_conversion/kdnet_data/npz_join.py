from __future__ import print_function
import re
import os
import numpy as np
import h5py

def find_regex_files(regex, directory):
    return [os.path.join(directory,file) for file in os.listdir(directory) if re.match(regex, file)]

def delete_files(files):
    for file in files:
        cmd = "rm {}".format(file)
        os.system(cmd)

def join_npz(directory, regex, output):
    files = find_regex_files(regex, directory)
    dict = {}
    for file in files:
        arch = np.load(file)
        for key in arch.keys():
            if key not in dict:
                dict[key] = []
            dict[key].append(arch[key])
    bigdict = {}
    for key in dict.keys():
        arr = np.concatenate(dict[key])
        bigdict[key] = arr
    if files:
        np.savez(output, **bigdict)
        delete_files(files)
    
def join_h5(directory, regex, output):
    files = find_regex_files(regex, directory)
    dict = {}
    test_nFaces = [0]
    train_nFaces = [0]
    test_offset = 0
    train_offset = 0
    for file in files:
        print('loading ' + file)
        arch = h5py.File(file, 'r')
        for key in arch.keys():
            if key == 'test_nnFaces':
                for n in arch[key]:
                    test_offset += n
                    test_nFaces.append(test_offset)
            elif key == 'train_nnFaces':
                for n in arch[key]:
                    train_offset += n
                    train_nFaces.append(train_offset)
            else:
                if key not in dict:
                    dict[key] = []
                dataset = arch.get(key)
                copy = np.copy(np.array(dataset))
                dict[key].append(copy)
        arch.close()
        
    dict['test_nFaces'] = test_nFaces  
    dict['train_nFaces'] = train_nFaces    
    print(test_nFaces)
    print(train_nFaces)
    if files:
        hf = h5py.File(os.path.join(directory, output), 'w')
        for key in dict.keys():
            try:
                arr = np.concatenate(dict[key])
            except ValueError:
                arr = np.array(dict[key])
            hf.create_dataset(key, data=arr)  
        
        hf.close()
        hf = h5py.File(os.path.join(directory, output), 'r')
        print(hf['test_nFaces'])
        hf.close()
        delete_files(files)
        
    return os.path.join(directory, output)
    
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="root directory of npz files to be joined")
    parser.add_argument("-re",  help="Regular expression")
    parser.add_argument("-o", type=str, help="name of the output file")
    parser.add_argument("-f", type=str,default='npz', help="npz or h5 format")

    args = parser.parse_args()
    if args.f == 'npz':
        join_npz(regex, args.o)
    elif args.f == 'h5':
        join_h5(regex, args.o)