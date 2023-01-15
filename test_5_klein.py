##########################################################
############### Klein rotation experiments ###############
##########################################################

import conjtest as ct
import numpy as np
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


def experiment_klein_diff_sp_grid():
    dl = 8

    rotations = np.array([[np.sqrt(2) / 10., (np.sqrt(3)) / 10.]])
    for embv in [2, 3, 4, 5]:
        data = generate_klein_rotation_data_test(n=8000,
                                                 rotations=rotations,
                                                 starting_points=np.array([[0., 0.], [0.3, 0.1], [0.2, 0.25], [0.1, 0.4]]),
                                                 delay=dl,
                                                 dims=[embv])
        kv = [5]
        # tv = list(range(100))
        tv = list(np.arange(1, 11, 2))  + [12, 15, 20, 25, 30, 40, 50] # +  list(np.arange(30, 81, 10))
        keys = [k for k in data.keys()]

        conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
        neigh_conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
        homeo = ct.embedding_homeomorphisms(data, 'klein', dl=dl)

        rev = True

        if rev:
            k2 = keys[0]
            ts2 = data[k2]
        else:
            k1 = keys[0]
            ts1 = data[k1]

        for j in range(2, len(keys)):
            if rev:
                k1 = keys[j]
                ts1 = data[k1]
                if k1[0] == 'emb':
                    base_name = output_directory + '/' + 'rev_klein_stpts_grid_dmax_t_' + k1[0] + str(k1[2]) + str(k1[3]) + '_vs_' + k2[0] + str(k2[1])
                elif embv == 2 and k1[0] == 'klein':
                    base_name = output_directory + '/' + 'rev_klein_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
                else:
                    base_name = 'what?'
                    continue
            else:
                k2 = keys[j]
                ts2 = data[k2]
                if k2[0] == 'emb':
                    base_name = output_directory + '/' + 'klein_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[3]) + str(k2[1])
                else:
                    base_name = output_directory + '/' + 'klein_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
            print(k1, k2)
            # ts2 = data[k2]
            new_n = min(len(ts1), len(ts2))

            print(base_name)
            # conj_diffs[j - 1, :, :] = ct.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv, t=tv,
            #                                             dist_fun='max')
            # conj_df = pd.DataFrame(data=conj_diffs[j - 1, :, :], index=[str(k) for k in kv], columns=[str(t) for t in tv])
            # conj_df.to_csv(output_directory + '/' + base_name + '_conj.csv')
            # print(conj_df.to_markdown())

            neigh_conj_diffs[j - 1, :, :] = ct.neigh_conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv,
                                                                    t=tv, dist_fun='max')
            neigh_conj_df = pd.DataFrame(data=neigh_conj_diffs[j - 1, :, :], index=[str(k) for k in kv],
                                         columns=[str(t) for t in tv])
            neigh_conj_df.to_csv(output_directory + '/' + base_name + '_neigh_conj.csv')
            print(neigh_conj_df.to_markdown())


def experiment_klein_embedding():
    base_name = output_directory + '/' + 'klein_dim_embedding'
    do_fnn = 0
    do_knn = 0
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
    labels = [str(k) for k in keys]
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
            fnn1, fnn2 = ct.fnn(ts1[:new_n], ts2[:new_n], r=rv, dist_fun='max')
            fnn_diffs[i-1, 0, :] = fnn1
            fnn_diffs[i-1, 1, :] = fnn2
        if do_knn:
            knn1, knn2 = ct.conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv, dist_fun='max')
            knn_diffs[i-1, 0, :] = knn1
            knn_diffs[i-1, 1, :] = knn2
        if do_conj:
            tsA = ts1[:new_n]
            tsB = ts2[:new_n]
            neigh_conj_diffs_t[i-1, 0, :] = ct.neigh_conjugacy_test(tsA, tsB, homeo(k1, k2, ts1, ts2), k=[5], t=tv,
                                                                    dist_fun='max')
            # neigh_conj_diffs_t[i-1, 1, :] = ct.neigh_conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=[5], t=tv,
            #                                                         dist_fun='max')
            neigh_conj_diffs_k[i-1, 0, :] = ct.neigh_conjugacy_test(tsA, tsB, homeo(k1, k2, ts1, ts2), k=kv, t=[10],
                                                                    dist_fun='max')[:, 0]
            # neigh_conj_diffs_k[i-1, 1, :] = ct.neigh_conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=kv, t=[10],
            #                                                         dist_fun='max')[:, 0]

    if do_fnn:
        fnn_df = pd.DataFrame(data=fnn_diffs[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(r) for r in rv])
        fnn_df.to_csv(output_directory + '/' + base_name + '_fnn_dir0' + '.csv')
        print(fnn_df.to_markdown())
        fnn_df = pd.DataFrame(data=fnn_diffs[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(r) for r in rv])
        fnn_df.to_csv(output_directory + '/' + base_name + '_fnn_dir1' + '.csv')
        print(fnn_df.to_markdown())
    if do_knn:
        knn_df = pd.DataFrame(data=knn_diffs[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        knn_df.to_csv(output_directory + '/' + base_name + '_knn_dir0' + '.csv')
        print(knn_df.to_markdown())
        knn_df = pd.DataFrame(data=knn_diffs[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        knn_df.to_csv(output_directory + '/' + base_name + '_knn_dir1' + '.csv')
        print(knn_df.to_markdown())
    if do_conj:
        conj_df = pd.DataFrame(data=neigh_conj_diffs_k[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        conj_df.to_csv(output_directory + '/' + base_name + '_conj_k_dir0' + '.csv')
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_k[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        conj_df.to_csv(output_directory + '/' + base_name + '_conj_k_dir1' + '.csv')
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_t[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(t) for t in tv])
        conj_df.to_csv(output_directory + '/' + base_name + '_conj_t_dir0' + '.csv')
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_t[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(t) for t in tv])
        conj_df.to_csv(output_directory + '/' + base_name + '_conj_t_dir1' + '.csv')
        print(conj_df.to_markdown())

