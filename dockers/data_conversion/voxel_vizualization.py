from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    path = "D:\\workspace\\diplomka\\data\\modelnet40_rot_test.npz"
    #path = "D:\\workspace\\diplomka\\data\\test.npz"
    
    parser.add_argument("-v", default=32, type=int, help="Resolution of the voxel grid")
    parser.add_argument("-f", default= path, type=str, help="npz file to load")
    parser.add_argument("-r", default = 24, type=int, help="Number of rotations of model along vertical axis")
    args = parser.parse_args()

    
    xt = np.asarray(np.load(args.f)['features'],dtype=np.float32)
    yt = np.asarray(np.load(args.f)['targets'],dtype=np.float32)
    N = args.v
    index = 0
    view = 0
        
    ma = xt[index+view].reshape((N,N,N))
    print(np.count_nonzero(xt[0] - xt[1]))
    
    print(xt.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axis("off")
    ax.set_aspect('equal')
    
    ax.voxels(ma,edgecolors='b')
    
    plt.show()