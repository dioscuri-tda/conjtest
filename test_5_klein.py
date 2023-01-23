##########################################################
############### Klein rotation experiments ###############
##########################################################

import conjtest as ct
import numpy as np
import os
import pandas as pd

output_directory = 'outputs/klein'
basename = 'klein'


def f_label(x):
    return str(np.round(x, 3))


def generate_klein_rotation_data_test(n=2000, starting_points=None, rotations=None, delay=8, dims=None):
    """
    @param n:
    @param starting_points:
    @param rotations:
    """
    if starting_points is None:
        starting_points = np.array([0., 0.])
    if dims is None:
        dims = [4]
    if rotations is None:
        rotations = np.array([[0.01, 0.01]])

    # parameters of the klein bottle
    kr = 1.
    kp = 8.
    ke = 1. / 2.

    data = {}
    for isp, sp in enumerate(starting_points):
        for rot in rotations:
            # print(sp, f_label(rot))
            tor = ct.torus_rotation_interval(n, steps=rot, starting_point=sp) * 2 * np.pi
            A = np.array([[1 / 4., 1 / 4., 1 / 4., 1 / 4.], [1 / 4., 1 / 4., 1 / 4., -1 / 4.],
                          [1 / 4., 1 / 4., -1 / 4., -1 / 4.], [1 / 4., -1 / 4., -1 / 4., -1 / 4.]])
            cos0d2 = np.cos(tor[:, 0] / 2.)
            sin0d2 = np.sin(tor[:, 0] / 2.)
            cos0 = np.cos(tor[:, 0])
            sin0 = np.sin(tor[:, 0])
            sin1 = np.sin(tor[:, 1])
            cos1 = np.cos(tor[:, 1])
            sin1m2 = np.sin(tor[:, 1] * 2.)
            klein = np.array([kr * (cos0d2 * cos1 - sin0d2 * sin1m2),
                              kr * (sin0d2 * cos1 - cos0d2 * sin1m2),
                              kp * cos0 * (1 + ke * sin1),
                              kp * sin0 * (1 + ke * sin1)]).transpose().reshape((len(tor), 4))
            shifted_klein = np.dot(A, klein.transpose()).transpose()
            data[('klein', f_label(sp), f_label(rot[0]), f_label(rot[1]))] = shifted_klein
            for d in dims:
                emb = ct.embedding(shifted_klein[:, 0].reshape((n,)), d, delay)
                data[('emb', 0, d, f_label(sp), f_label(rot[0]), f_label(rot[1]))] = emb
    return data


def experiment_klein_embedding():
    """
    Experiment 5A described in Section 5.1.1. of https://arxiv.org/abs/2301.06753
    """
    base_name = output_directory + '/' + 'klein_dim_embedding'
    do_fnn = 1
    do_knn = 1
    do_conj = 1

    dl = 8
    tv = list(np.arange(1, 21, 4)) + list(np.arange(25, 51, 5))
    kv = list(np.arange(1, 21, 1)) + list(np.arange(25, 51, 5))
    rv = list(np.arange(1, 21, 1))
    dimsv = [1, 2, 3, 4, 5, 6, 7, 8]
    fnn_diffs = np.zeros((len(dimsv)-1, 2, len(rv)))
    knn_diffs = np.zeros((len(dimsv)-1, 2, len(kv)))
    neigh_conj_diffs_k = np.zeros((len(dimsv)-1, 2, len(kv)))
    neigh_conj_diffs_t = np.zeros((len(dimsv)-1, 2, len(tv)))

    rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(3)) / 10.]])
    data = generate_klein_rotation_data_test(n=8000,
                                             rotations=rotations,
                                             starting_points=np.array([[0., 0.]]),
                                             delay=dl,
                                             dims=dimsv)

    homeo = ct.embedding_homeomorphisms(data, 'klein', dl=dl)

    keys = [k for k in data.keys()]
    print(data.keys())

    pairs = [(i, i+1) for i in dimsv[:-1]]
    for (i, j) in pairs:
        k1 = keys[i]
        k2 = keys[j]
        print(k1, ' vs. ', k2)
        if len(data[k1].shape) == 1:
            ts1 = data[k1].reshape((len(data[k1]), 1))
        else:
            ts1 = data[k1]
        if len(data[k2].shape) == 1:
            ts2 = data[k2].reshape((len(data[k2]), 1))
        else:
            ts2 = data[k2]
        new_n = min(len(ts1), len(ts2))
        if do_fnn:
            fnn1, fnn2 = ct.fnn_conjugacy_test(ts1[:new_n], ts2[:new_n], r=rv, dist_fun='max')
            fnn_diffs[i-1, 0, :] = fnn1
            fnn_diffs[i-1, 1, :] = fnn2
        if do_knn:
            knn1, knn2 = ct.knn_conjugacy_test(ts1[:new_n], ts2[:new_n], k=kv, dist_fun='max')
            knn_diffs[i-1, 0, :] = knn1
            knn_diffs[i-1, 1, :] = knn2
        if do_conj:
            tsA = ts1[:new_n]
            tsB = ts2[:new_n]
            neigh_conj_diffs_t[i-1, 0, :] = ct.conjtest_plus(tsA, tsB, homeo(k1, k2, ts1, ts2), k=[5], t=tv,
                                                             dist_fun='max')
            neigh_conj_diffs_t[i-1, 1, :] = ct.conjtest_plus(tsB, tsA, homeo(k2, k1, ts2, ts1), k=[5], t=tv,
                                                             dist_fun='max')
            neigh_conj_diffs_k[i-1, 0, :] = ct.conjtest_plus(tsA, tsB, homeo(k1, k2, ts1, ts2), k=kv, t=[10],
                                                             dist_fun='max')[:, 0]
            neigh_conj_diffs_k[i-1, 1, :] = ct.conjtest_plus(tsB, tsA, homeo(k2, k1, ts2, ts1), k=kv, t=[10],
                                                             dist_fun='max')[:, 0]

    if do_fnn:
        fnn_df = pd.DataFrame(data=fnn_diffs[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(r) for r in rv])
        fnn_df.to_csv(output_directory + '/' + base_name + '_fnn_dir0' + '.csv')
        print('----------------------------------------------------------------------------------')
        print("FNN; column: value of r; row: dimension")
        print("time series 1 vs. time series 2")
        print(fnn_df.to_markdown())
        fnn_df = pd.DataFrame(data=fnn_diffs[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(r) for r in rv])
        fnn_df.to_csv(output_directory + '/' + base_name + '_fnn_dir1' + '.csv')
        print("time series 2 vs. time series 1")
        print(fnn_df.to_markdown())
    if do_knn:
        knn_df = pd.DataFrame(data=knn_diffs[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        knn_df.to_csv(output_directory + '/' + base_name + '_knn_dir0' + '.csv')
        print('----------------------------------------------------------------------------------')
        print("KNN; column: value of k; row: dimension")
        print("time series 1 vs. time series 2")
        print(knn_df.to_markdown())
        knn_df = pd.DataFrame(data=knn_diffs[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        knn_df.to_csv(output_directory + '/' + base_name + '_knn_dir1' + '.csv')
        print("time series 2 vs. time series 1")
        print(knn_df.to_markdown())
    if do_conj:
        conj_df = pd.DataFrame(data=neigh_conj_diffs_k[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        conj_df.to_csv(output_directory + '/' + base_name + '_conj_k_dir0' + '.csv')
        print('----------------------------------------------------------------------------------')
        print('ConjTestPlus; column: value of k; row: dimension')
        print("time series 1 vs. time series 2")
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_k[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        conj_df.to_csv(output_directory + '/' + base_name + '_conj_k_dir1' + '.csv')
        print("time series 2 vs. time series 1")
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_t[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(t) for t in tv])
        conj_df.to_csv(output_directory + '/' + base_name + '_conj_t_dir0' + '.csv')
        print('----------------------------------------------------------------------------------')
        print('ConjTestPlus; column: value of t; row: dimension')
        print("time series 1 vs. time series 2")
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_t[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(t) for t in tv])
        conj_df.to_csv(output_directory + '/' + base_name + '_conj_t_dir1' + '.csv')
        print("time series 2 vs. time series 1")
        print(conj_df.to_markdown())

if __name__ == '__main__':
    os.makedirs(output_directory, exist_ok=True)

    ### experiment 5A - 5.1.1
    experiment_klein_embedding()

