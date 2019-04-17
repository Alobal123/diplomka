import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
   
    path = "D:\\workspace\\diplomka\\dockers\\data_conversion\\test_0.h5"
    #path = ".\\train_0.h5"
    parser.add_argument("-f", default= path, type=str, help="npz file to load")
    args = parser.parse_args()
    
    f = h5py.File(args.f)
    index = 1
    labels = np.array(f['label'])
    points = np.array(f['data'])
    print(points.shape)
    #soms = np.array(f['som'])
    fig = plt.figure()
    
    ax = Axes3D(fig)
    #ax.axis("off")
    ax.scatter(points[index,:,0], points[index,:,1],points[index,:,2])
    #ax.scatter(soms[index,:,0], soms[index,:,1],soms[index,:,2])
    plt.show()
    