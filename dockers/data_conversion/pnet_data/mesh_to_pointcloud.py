from __future__ import print_function
import numpy as np
from mesh_files import *
import point_cloud_utils as pcu

def triangle_normal(v1,v2,v3):
    return np.cross(v2-v1, v3-v1)

def traingle_area(v1, v2, v3):
    """
    >>> traingle_area(np.array([0,0,0]), np.array([0,1,0]), np.array([0,0,1]))
    0.5
    >>> traingle_area(np.array([0,0,0]), np.array([0,0,-9]), np.array([-9,0,0]))
    40.5
    >>> traingle_area(np.array([0,0,0]), np.array([1,0,0]), np.array([0,2,0]))
    1.0
    >>> traingle_area(np.array([0,0,0]), np.array([-18,0,0]), np.array([0,-1,0]))
    9.0
    """
    return 0.5 * np.linalg.norm(triangle_normal(v1, v2, v3))

def normalize(vector):
    return vector / np.sum(vector)


    

def mesh_to_point_cloud(points, triangles, n, normal=False):
    """
    test the distribution of the sampled points between two triangles
    >>> points = np.array([[0,0,0], [1,0,0], [0,2,0], [-18, 0, 0], [0, -1, 0]])
    
    >>> triangles = np.array([[0,1,2],[0,3,4]])
    
    >>> sample = mesh_to_point_cloud(points, triangles, 1000)
    
    >>> sum([1 for p in sample if p[0]<0]) in range(850,950)
    True
    
    test overall distribution of the points
    >>> points = np.array([[0,0,0], [1,0,0], [0,1,0], [1, 1, 0]])
    
    >>> triangles = np.array([[0,1,2],[1,2,3]])
    
    >>> sample = mesh_to_point_cloud(points, triangles, 4000)
    
    >>> sum([1 for point in sample if (np.array(point) <= np.array([0.5,0.5,0])).all() and (np.array(point) >= np.array([0,0,0])).all()]) in range(950,1050)
    True
    
    """
    distribution = find_area_distribution(points, triangles)
    chosen = np.random.choice(range(len(triangles)),size=n, p=distribution)
    chosen_points = points[triangles[chosen]]
    normals = [triangle_normal(*triangle) for triangle in chosen_points] if normal else None
    u = np.random.rand(n,1)
    v = np.random.rand(n,1)
    is_outside = u + v > 1
    u[is_outside] = 1 - u[is_outside]
    v[is_outside] = 1 - v[is_outside]
    w = 1 - (u+v)
    xs = chosen_points[:, 0 ,:] * u
    ys = chosen_points [:, 1, :] * v
    zs = chosen_points[:, 2, :] * w
    
    if normals: 
        return np.concatenate(((xs+ys+zs),normals), axis=1)
    else:
        return (xs + ys + zs)

def find_area_distribution(points, triangles):
    """
    >>> points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,-9,0], [0,0,-1]])
    
    >>> triangles = [[0,1,2],[0,3,4]]
    
    >>> find_area_distribution(points, triangles)
    array([0.1, 0.9])
    """
    distribution = np.zeros((len(triangles)))
    for t in range(len(triangles)):
        triangle = triangles[t]
        v1,v2,v3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        distribution[t] = traingle_area(v1, v2, v3)
    return normalize(distribution)
       

def file_to_pointcloud(filename, type, args):
    if type == 'shapenet':
        points, triangles, quads = read_obj_file(filename)
    elif type == 'modelnet':
        points, triangles, quads = read_off_file(filename)
    else:
        print("bad dataset type")
    if args.normal:
        return mesh_to_point_cloud(points, triangles, args.num_points, normal=True)
    else:
        if args.lloyd:
            return pcu.sample_mesh_lloyd(points, triangles, args.num_points)
        return mesh_to_point_cloud(points, triangles, args.num_points)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    