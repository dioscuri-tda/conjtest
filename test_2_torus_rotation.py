##########################################################
############### Torus rotation experiments ###############
##########################################################

import conjtest as ct
import numpy as np
import numba
import os

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
            points = ct.torus_rotation_interval(n, steps=rot, starting_point=sp)
            data[('torus_rot', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = points
            data[('torus_proj_x', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = points[:, 0]
    return data


@numba.jit(nopython=True, fastmath=True)
def torus_max_dist(x, y):
    dist = 0.
    modx = np.mod(x - y, 1.)
    mody = np.mod(y - x, 1.)
    for i in range(len(x)):
        di = min(modx[i], mody[i])
        dist = max(di, dist)
    return dist


def vanilla_experiment_torus():
    """
    Experiment 2A described in Section 4.2.1. of https://arxiv.org/abs/2301.06753
    """
    n = 2000
    starting_points = np.array([[0., 0.], [0.1, 0.]])

    rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(3)) / 10.],
                          [(1.1 * np.sqrt(2)) / 10., (np.sqrt(3)) / 10.],
                          [np.sqrt(3) / 10., (np.sqrt(3)) / 10.]])
    data = generate_torus_rotation_data_test(n=n, starting_points=starting_points,
                                             rotations=rotations)

    pairs = [(0, i) for i in range(1, len(data.keys()))]

    def homeo(k1, k2, ts1, _):
        if k1[0] == k2[0]:
            return lambda x: x
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

    rv = [2]
    kv = [5]
    tv = [5]

    ### for computing all vs. all set
    # pairs = None
    ct.vanilla_experiment(data, basename + '_torus_n' + str(n), kv, tv, rv, True, True, homeo,
                          pairs=pairs, dist_fun=torus_max_dist, out_dir=output_directory)


if __name__ == '__main__':
    os.makedirs(output_directory, exist_ok=True)

    ### experiment 2A - 4.1.1
    vanilla_experiment_torus()
