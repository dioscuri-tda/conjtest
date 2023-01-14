import conjtest as ct
import numba
import numpy as np
import os
import pandas as pd

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
    np.random.seed(0)
    perturbations = (np.random.random(len(x)) * 2 * pl) - pl
    return np.mod(x + perturbations, 1.)


def angle_power_interval(x, p):
    return np.power(x, p)


# identity map
def id(x):
    return x


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
            crc_points = ct.circle_rotation_interval(n, step=r, starting_point=sp)
            data[(f_label(sp), f_label(r))] = crc_points
    if nonlin_params is not None:
        for s in nonlin_params:
            sp = starting_points[0]
            r = rotations[0]
            data[(f_label(sp), f_label(r), s)] = ct.circle_rotation_interval(n, step=r, starting_point=sp, nonlin=s)
    return data


def vanilla_experiment_rotation():
    """
    Experiment 1A described in Section 4.1.1. of [link]
    """
    n = 2000
    starting_points = np.array([0., 0.25])
    rotations = [np.sqrt(2) / 10., (np.sqrt(2) + 0.2) / 10., 2 * (np.sqrt(2)) / 10.]
    data = generate_circle_rotation_data_test(n=n, starting_points=starting_points,
                                              rotations=rotations, nonlin_params=[2.])

    print(data.keys())
    pert_key = ('0.0', '0.141', 2.0)
    data[pert_key + ('0.05',)] = perturbation_on_circle_interval(data[pert_key], 0.05)
    pairs = [(0, i) for i in range(1, len(data.keys()))]
    print(pairs)

    def homeo(k1, k2, ts1, ts2):
        if len(k1) == len(k2):
            return id
        elif len(k1) == 2 and len(k2) > 2:
            return lambda x: angle_power_interval(x, 1. / k2[2])
        elif len(k1) > 2 and len(k2) == 2:
            return lambda x: angle_power_interval(x, k1[2])

    kv = [3, 5]
    tv = [5, 10]
    rv = [2, 3]
    ### for computing all vs. all set
    # pairs = None
    ct.vanilla_experiment(data, basename + '_rotation_n' + str(n), kv, tv, rv, do_knn=True, do_conj=True, homeo=homeo,
                          pairs=pairs, dist_fun=sphere_max_dist, out_dir=output_directory)


def experiment_rotation_int_rv_grid():
    """
    Experiment 1B described in Section 4.1.2. of [link]
    """
    # rotations = np.arange(0.05, 0.751, 0.025)
    npoints = 2000
    base_angle = np.sqrt(2) / 10
    nsteps = 50
    nsteps = 2
    step = base_angle / (2 * nsteps)
    rotations = [base_angle + (step * (i - nsteps)) for i in range(int(nsteps * 3.5))]
    total_steps = nsteps * 3.5
    print(np.round(rotations, 4))
    data = generate_circle_rotation_data_test(n=npoints, rotations=rotations)
    kv = [1, 5, 10, 20]
    rv = [1, 2, 3, 8]
    tv = [1, 3, 5, 10]
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]

    conj_diffs = np.zeros((len(keys), len(kv), len(tv)))
    knns_diffs = np.zeros((len(keys), len(kv), 2))
    fnns_diffs = np.zeros((len(keys), len(rv), 2))

    k1 = keys[int(nsteps)]
    base_name = 'rotation_int_grid_' + str(int(total_steps)) + '_r'
    for j in range(0, len(keys)):
        k2 = keys[j]
        print(k1, k2)
        if len(data[k1].shape) == 1:
            ts1 = data[k1].reshape((len(data[k1]), 1))
        else:
            ts1 = data[k1]
        if len(data[k2].shape) == 1:
            ts2 = data[k2].reshape((len(data[k2]), 1))
        else:
            ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))
        conj_diffs[j, :, :] = ct.conjugacy_test(ts1[:new_n], ts2[:new_n], id, k=kv, t=tv, dist_fun=sphere_max_dist)
        knn1, knn2 = ct.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv, dist_fun=sphere_max_dist)
        knns_diffs[j, :, 0] = knn1
        knns_diffs[j, :, 1] = knn2
        fnn1, fnn2 = ct.fnn(ts1[:new_n], ts2[:new_n], r=rv, dist_fun=sphere_max_dist)
        fnns_diffs[j, :, 0] = fnn1
        fnns_diffs[j, :, 1] = fnn2

    for it, t in enumerate(tv):
        conj_df = pd.DataFrame(data=np.transpose(conj_diffs[:, :, it]), index=[str(k) for k in kv],
                               columns=[f_label(r) for r in rotations])
        strt = str(t) if t > 9 else '0' + str(t)
        conj_df.to_csv(output_directory + '/' + base_name + '_t' + strt + '_n' + str(npoints) + '.csv')
        print('----------------------------------------------------------------------------------')
        print('ConjTest - t: ' + str(t) + '; column: value of k; row: angle')
        print(conj_df.to_markdown())

    knns1_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 0]), index=[str(k) for k in kv],
                            columns=[f_label(r) for r in rotations])
    knns1_df.to_csv(output_directory + '/' + base_name + '_knn1' + '_n' + str(npoints) + '.csv')
    knns2_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 1]), index=[str(k) for k in kv],
                            columns=[f_label(r) for r in rotations])
    knns2_df.to_csv(output_directory + '/' + base_name + '_knn2' + '_n' + str(npoints) + '.csv')
    print('----------------------------------------------------------------------------------')
    print("KNN; column: value of k; row: angle")
    print(knns1_df.to_markdown())
    print(knns2_df.to_markdown())

    fnns1_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 0]), index=[str(r) for r in rv],
                            columns=[f_label(r) for r in rotations])
    fnns1_df.to_csv(output_directory + '/' + base_name + '_fnn1' + '_n' + str(npoints) + '.csv')
    fnns2_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 1]), index=[str(r) for r in rv],
                            columns=[f_label(r) for r in rotations])
    fnns2_df.to_csv(output_directory + '/' + base_name + '_fnn2' + '_n' + str(npoints) + '.csv')
    print('----------------------------------------------------------------------------------')
    print("FNN; column: value of r; row: angle")
    print(fnns1_df.to_markdown())
    print(fnns2_df.to_markdown())


def experiment_rotation_noise_grid():
    """
    Experiment 1C described in Section 4.1.3. of [link]
    """
    npoints = 2000
    base_angle = np.sqrt(2) / 10
    power_param = 2.
    data = generate_circle_rotation_data_test(n=npoints, rotations=[base_angle], nonlin_params=[power_param])
    kv = [1, 3, 5, 10]
    rv = [2, 5, 8]
    tv = [5, 10]
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]

    perturbation_levels = np.arange(0.00, 0.2501, 0.01)
    nkey = keys[-1]
    nnkey = nkey + (f_label(0.00),)
    data[nnkey] = data[nkey]
    data.pop(nkey)
    for ip, p, in enumerate(perturbation_levels):
        np.random.seed(0)
        data[nkey + (f_label(p),)] = perturbation_on_circle_interval(data[nnkey], p)

    keys = [k for k in data.keys()]
    column_labels = [k[-1] for k in keys[1:]]

    conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
    knns_diffs = np.zeros((len(keys) - 1, len(kv), 2))
    fnns_diffs = np.zeros((len(keys) - 1, len(rv), 2))

    def homeo(x):
        return angle_power_interval(x, 1. / power_param)

    k1 = keys[0]
    base_name = 'rotation_grid_dmax_noise'
    for j in range(1, len(keys)):
        k2 = keys[j]
        print(k1, k2)
        if len(data[k1].shape) == 1:
            ts1 = data[k1].reshape((len(data[k1]), 1))
        else:
            ts1 = data[k1]
        if len(data[k2].shape) == 1:
            ts2 = data[k2].reshape((len(data[k2]), 1))
        else:
            ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))
        conj_diffs[j - 1, :, :] = ct.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo, k=kv, t=tv,
                                                    dist_fun=sphere_max_dist)
        knn1, knn2 = ct.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv, dist_fun=sphere_max_dist)
        knns_diffs[j - 1, :, 0] = knn1
        knns_diffs[j - 1, :, 1] = knn2
        fnn1, fnn2 = ct.fnn(ts1[:new_n], ts2[:new_n], r=rv, dist_fun=sphere_max_dist)
        fnns_diffs[j - 1, :, 0] = fnn1
        fnns_diffs[j - 1, :, 1] = fnn2

    for it, t in enumerate(tv):
        conj_df = pd.DataFrame(data=np.transpose(conj_diffs[:, :, it]), index=[str(k) for k in kv],
                               columns=column_labels)
        strt = str(t) if t > 9 else '0' + str(t)
        conj_df.to_csv(output_directory + '/' + base_name + '_t' + strt + '_n' + str(npoints) + '.csv')
        print('----------------------------------------------------------------------------------')
        print('ConjTest - t: ' + str(t) + '; column: value of k; row: noise level')
        print(conj_df.to_markdown())

    knns1_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 0]), index=[str(k) for k in kv],
                            columns=column_labels)
    knns1_df.to_csv(output_directory + '/' + base_name + '_knn1' + '_n' + str(npoints) + '.csv')
    knns2_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 1]), index=[str(k) for k in kv],
                            columns=column_labels)
    knns2_df.to_csv(output_directory + '/' + base_name + '_knn2' + '_n' + str(npoints) + '.csv')
    print('----------------------------------------------------------------------------------')
    print("KNN; column: value of k; row: noise level")
    print(knns1_df.to_markdown())
    print(knns2_df.to_markdown())

    fnns1_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 0]), index=[str(r) for r in rv],
                            columns=column_labels)
    fnns1_df.to_csv(output_directory + '/' + base_name + '_fnn1' + '_n' + str(npoints) + '.csv')
    fnns2_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 1]), index=[str(r) for r in rv],
                            columns=column_labels)
    fnns2_df.to_csv(output_directory + '/' + base_name + '_fnn2' + '_n' + str(npoints) + '.csv')
    print('----------------------------------------------------------------------------------')
    print("fNN; column: value of r; row: noise level")
    print(fnns1_df.to_markdown())
    print(fnns2_df.to_markdown())


if __name__ == '__main__':
    os.makedirs(output_directory, exist_ok=True)
    # Experiment 1A - 4.1.1
    vanilla_experiment_rotation()
    # Experiment 1B - 4.1.2
    # experiment_rotation_int_rv_grid()
    # Experiment 1C - 4.1.3
    # experiment_rotation_noise_grid()
