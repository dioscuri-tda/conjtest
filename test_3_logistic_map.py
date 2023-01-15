########################################################
############### Logistic map experiments ###############
########################################################
import conjtest as ct
import numpy as np
import pandas as pd

output_directory = 'outputs/logistic'
basename = 'logistic'


# homeomorphism from a logistic map to a tent map
def h_log_to_tent(x):
    return 2 * np.arcsin(np.sqrt(x)) / np.pi
    # return 2 * sympy.asin(sympy.sqrt(x)) / sympy.pi


# homeomorphism from a tent map to a logistic map
def h_tent_to_log(x):
    # return sympy.sin(sympy.pi * x / 2) ** 2
    return np.sin(np.pi * x / 2) ** 2


def f_label(x):
    return str(np.round(x, 3))


def generate_log_tent_data(n=500, starting_points=None, log_parameters=None):
    # starting point
    # starting_points = [sympy.Rational(20, 100), sympy.Rational(21, 100), sympy.Rational(25, 100)]
    # parameters = [sympy.Rational(4, 1)]
    if starting_points is None:
        starting_points = [0.2]
    if log_parameters is None:
        log_parameters = [4.]

    data = {}
    for p in log_parameters:
        for sp in starting_points:
            print((sp, p))
            logi = np.array(ct.logistic_map(n, r=p, starting_point=sp))

            if p == 4.0:
                # tent = np.array(ct.logistic_map(n, r=4.00, starting_point=sp))
                # tent = np.array([h_log_to_tent(x).evalf() for x in logi], dtype=float)
                tent = np.array([h_log_to_tent(x) for x in logi], dtype=float)
                # logi = np.array([x.evalf() for x in logi], dtype=float)
                data[("tent", p, sp)] = tent

            data[("logm", p, sp)] = logi
    return data


def experiment_log_tent():
    np.random.seed(0)
    n = 2000

    # FNN, KNN, conjTest experiment
    # data = generate_log_tent_data(n, starting_points=[0.2, 0.201, 0.3], log_parameters=[4.0, 3.999, 3.99, 3.8])
    # data = generate_log_tent_data(n, starting_points=[0.2, 0.201, 0.3], log_parameters=[4.0, 3.99, 3.9, 3.8])
    # data = generate_log_tent_data(n, starting_points=[0.2], log_parameters=[4.0, 3.999, 3.8])
    data = generate_log_tent_data(n, starting_points=[0.2, 0.21], log_parameters=[4.0, 3.99])
    # data = generate_log_tent_data(n, starting_points=[0.2], log_parameters=[4.0])
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]
    print(data.keys())
    kv = [3, 5]
    # kv = [10]
    tv = [5, 10]
    rv = [2, 3]

    pairs = [(1, i) for i in range(0, len(data.keys()))]
    # pairs = [(1,0), (1, 4), (1, 2), (1, 5)]
    # pairs = [(1,4)]
    def homeo(k1, k2, ts1, ts2):
        if k1[0] == 'logm' and k2[0] == 'tent':
            return h_log_to_tent
        elif k1[0] == 'tent' and k2[0] == 'logm':
            return h_tent_to_log
        else:
            return id

    ct.vanilla_experiment(data, basename + '_log_tent_n' + str(n), kv, tv, rv, True, True, homeo, pairs=pairs)


def experiment_log_rv_grid():
    npoints = 2000
    log_params = np.arange(4., 3.8, -0.005)
    total_steps = len(log_params)
    data = generate_log_tent_data(n=2000, starting_points=[0.2], log_parameters=log_params)
    kv = [1, 3, 5, 10]
    rv = [2, 3, 5, 8]
    # tv = list(range(1, 20, 5))
    # tv = [1, 3, 5, 10, 15, 20]
    tv = [1, 3, 5, 10, 20]
    data.pop(list(data.keys())[0])
    keys = [k for k in data.keys()]

    conj_diffs = np.zeros((len(keys), len(kv), len(tv)))
    knns_diffs = np.zeros((len(keys), len(kv), 2))
    fnns_diffs = np.zeros((len(keys), len(rv), 2))

    k1 = keys[0]
    base_name = basename + '_log_params_grid_' + str(int(total_steps)) + '_'
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
        # conj_diffs[j, :, :] = ct.conjugacy_test(ts1[:new_n], ts2[:new_n], id, k=kv, t=tv)
        knn1, knn2 = ct.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv)
        knns_diffs[j, :, 0] = knn1
        knns_diffs[j, :, 1] = knn2
        fnn1, fnn2 = ct.fnn(ts1[:new_n], ts2[:new_n], r=rv)
        fnns_diffs[j, :, 0] = fnn1
        fnns_diffs[j, :, 1] = fnn2

    # for it, t in enumerate(tv):
    #     conj_df = pd.DataFrame(data=np.transpose(conj_diffs[:, :, it]), index=[str(k) for k in kv],
    #                            columns=[f_label(r) for r in log_params])
    #     strt = str(t) if t > 9 else '0' + str(t)
    #     conj_df.to_csv(output_directory + base_name + '_t' + strt + '_n' + str(npoints) + '.csv')
    #     print(conj_df.to_markdown())

    knns1_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 0]), index=[str(k) for k in kv],
                            columns=[f_label(r) for r in log_params])
    knns1_df.to_csv(output_directory + '/' +  base_name + '_knn1' + '_n' + str(npoints) + '.csv')
    knns2_df = pd.DataFrame(data=np.transpose(knns_diffs[:, :, 1]), index=[str(k) for k in kv],
                            columns=[f_label(r) for r in log_params])
    knns2_df.to_csv(output_directory + '/' + base_name + '_knn2' + '_n' + str(npoints) + '.csv')
    print(knns1_df.to_markdown())
    print(knns2_df.to_markdown())

    fnns1_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 0]), index=[str(r) for r in rv],
                            columns=[f_label(r) for r in log_params])
    fnns1_df.to_csv(output_directory + '/' + base_name + '_fnn1' + '_n' + str(npoints) + '.csv')
    fnns2_df = pd.DataFrame(data=np.transpose(fnns_diffs[:, :, 1]), index=[str(r) for r in rv],
                            columns=[f_label(r) for r in log_params])
    fnns2_df.to_csv(output_directory + '/' + base_name + '_fnn2' + '_n' + str(npoints) + '.csv')
