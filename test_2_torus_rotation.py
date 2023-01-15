##########################################################
############### Torus rotation experiments ###############
##########################################################

import conjtest as ct
import numpy as np
import numba

output_directory = 'outputs/rotation_torus'
basename = 'rotation_torus'

def f_label(x):
    return str(np.round(x, 3))


def generate_torus_rotation_data_test(n=2000, starting_points=None, rotations=None):
    """
    @param n:
    @param starting_points:
    @param rotations:
    """
    if starting_points is None:
        starting_points = np.array([0., 0.])
    if rotations is None:
        rotations = np.array([[0.01, 0.01]])

    data = {}
    for isp, sp in enumerate(starting_points):
        for rot in rotations:
            # print(sp, f_label(rot))
            points = ct.torus_rotation_interval(n, steps=rot, starting_point=sp)
            data[('torus_rot', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = points
            # if isp == 0:
            data[('torus_proj_x', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = points[:, 0]
            # data[('torus_proj_y', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = points[:, 1]
    return data


def experiment_torus():
    n = 2000
    # starting_points = np.array([[0., 0.]])
    starting_points = np.array([[0., 0.], [0.1, 0.]])

    # rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(2)) / 5.], [(np.sqrt(2)) / 10., (np.sqrt(3)) / 10.]])
    rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(3)) / 10.], [(1.1 * np.sqrt(2)) / 10., (np.sqrt(3)) / 10.],
                          [np.sqrt(3) / 10., (np.sqrt(3)) / 10.]])
    # rotations = np.array([[(np.sqrt(2)) / 10., (np.sqrt(3)) / 10.], [np.sqrt(3) / 10., (np.sqrt(3)) / 10.], [np.sqrt(2) / 5., (np.sqrt(3)) / 5.]])
    data = generate_torus_rotation_data_test(n=n, starting_points=starting_points,
                                             rotations=rotations)

    pairs = [(0, i) for i in range(1, len(data.keys()))]

    def homeo(k1, k2, ts1, ts2):
        if k1[0] == k2[0]:
            return id
        elif k1[0] == 'torus_rot' and k2[0] == 'torus_proj_x':
            return lambda x: x[:, 0]
        elif k1[0] == 'torus_rot' and k1[0] == 'torus_proj_y':
            return lambda x: x[:, 1]
        elif k2[0] == 'torus_rot' and k1[0] == 'torus_proj_x':
            return lambda x: np.hstack((x.reshape(len(x), 1), np.zeros((len(x), 1))))
        elif k2[0] == 'torus_rot' and k1[0] == 'torus_proj_y':
            return lambda x: np.hstack((x.reshape(len(x), 1), np.zeros((len(x), 1))))
        else:
            return None

    @numba.jit(nopython=True, fastmath=True)
    def torus_max_dist(x, y):
        dist = 0.
        modx = np.mod(x - y, 1.)
        mody = np.mod(y - x, 1.)
        for i in range(len(x)):
            di = min(modx[i], mody[i])
            dist = max(di, dist)
        return dist

    # @numba.jit(nopython=True, fastmath=True)
    # def torus_max_dist(x, y):
    #     return np.max(np.min(np.array([np.mod(x - y, 1.), np.mod(y - x, 1.)]), axis=0), axis=0)
    # if len(x.shape) == 1:
    #     return np.max(np.min(np.array([np.mod(x - y, 1.), np.mod(y - x, 1.)]), axis=0), axis=0)
    # elif len(x.shape) == 2:
    #     return np.max(np.min(np.array([np.mod(x - y, 1.), np.mod(y - x, 1.)]), axis=0), axis=1)
    # return np.max(np.min(np.array([np.mod(x - y, 1.), np.mod(y - x, 1.)]), axis=0), axis=1)
    # return np.min(np.array([np.mod(x-y, 1.), np.mod(y-x, 1.)]))
    # print(torus_max_dist(np.array([0.4, 0.3]), np.array([0.0, 0.8])))
    # print(torus_max_dist(np.array([0.4, 0.3]), np.array([0.5, 0.1])))

    kv = [1, 3, 5]
    tv = [1, 3, 5, 10]
    rv = [1, 2, 3]
    ct.vanilla_experiment(data, basename + '_torus_n' + str(n), kv, tv, rv, False, True, homeo,
                          pairs=pairs, dist_fun=torus_max_dist, out_dir=output_directory)
