import conjtest as ct
import numba
import numpy as np
import os

output_directory = 'outputs/rotation_circle'
basename = 'rotation_circle'

def f_label(x):
    return str(np.round(x, 3))


@numba.jit(nopython=True, fastmath=True)
def sphere_max_dist(x, y):
    modx = np.mod(x - y, 1.)
    mody = np.mod(y - x, 1.)
    return min(modx[0], mody[0])


def perturbation_on_circle_interval(x, pl):
    """
    @param x: points from interval [0,1] interpreted as a circle
    @param pl: perturbation_level
    @return:
    """
    perturbations = (np.random.random(len(x)) * 2 * pl) - pl
    return np.mod(x + perturbations, 1.)


def generate_circle_rotation_data_test(n=1000, starting_points=None, rotations=None, nonlin_params=None):
    """
    @param n:
    @param starting_points:
    @param rotations: a list of rotations to consider (in radians)
    """
    if starting_points is None:
        starting_points = np.array([0.])
    if rotations is None:
        rotations = [0.1]

    data = {}
    for sp in starting_points:
        for r in rotations:
            print(sp, f_label(r))
            crc_points = ct.circle_rotation_interval(n, step=r, starting_point=sp)
            print(crc_points.shape)
            data[(f_label(sp), f_label(r))] = crc_points
    if nonlin_params is not None:
        for s in nonlin_params:
            sp = starting_points[0]
            r = rotations[0]
            data[(f_label(sp), f_label(r), s)] = ct.circle_rotation_interval(n, step=r, starting_point=sp, nonlin=s)
    return data


def vanilla_experiment_rotation():
    n = 2000
    starting_points = np.array([0., 0.25])
    rotations = [np.sqrt(2) / 10., (np.sqrt(2) + 0.2) / 10., 2 * (np.sqrt(2)) / 10.]
    data = generate_circle_rotation_data_test(n=n, starting_points=starting_points,
                                              rotations=rotations, nonlin_params=[2.])

    last_key = list(data.keys())[-1]
    print(data.keys())
    pert_key = ('0.0', '0.141', 2.0)
    data[pert_key + ('0.05',)] = perturbation_on_circle_interval(data[pert_key], 0.05)
    pairs = [(0, i) for i in range(1, len(data.keys()))]
    print(pairs)

    def homeo(k1, k2, ts1, ts2):
        if len(k1) == len(k2):
            return id
        elif len(k1) == 2 and len(k2) > 2:
            return lambda x: np.power(x, 1. / k2[2])
        elif len(k1) > 2 and len(k2) == 2:
            return lambda x: np.power(x, k1[2])

    kv = [3, 5]
    tv = [5, 10]
    rv = [2, 3]
    ct.vanilla_experiment(data, basename + '_rotation_n' + str(n), kv, tv, rv, True, False, homeo, pairs=pairs,
                          dist_fun=sphere_max_dist, out_dir=output_directory)

if __name__ == '__main__':
    os.makedirs(output_directory, exist_ok=True)
    vanilla_experiment_rotation()
