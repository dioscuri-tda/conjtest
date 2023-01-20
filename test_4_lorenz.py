#########################################################
############### Lorenz system experiments ###############
#########################################################

import conjtest as ct
import numpy as np
import os
import pandas as pd

output_directory = 'outputs/lorenz'
basename = 'lorenz'


def f_label(x):
    return str(np.round(x, 3))


def generate_lorenz_data_starting_test(n=5000, starting_points=None, delay=5, emb=True, emb_dim=3):
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)
    data = {}
    for idx, sp in enumerate(starting_points[:]):
        lorenz = ct.lorenz_attractor(n, starting_point=sp, skip=2000)
        data[("lorenz", tuple(sp))] = lorenz
        if emb:
            lemb = ct.embedding(lorenz[:, 0].reshape((n,)), emb_dim, dl)
            data[("emb", 0, emb_dim, tuple(sp))] = lemb
    return data


def generate_lorenz_data_embeddings_test(n=5000, starting_points=None, delay=5, dims=None, axis=None):
    if dims is None:
        dims = [3]
    if axis is None:
        axis = [0]
    if starting_points is None:
        starting_points = [[1., 1., 1.]]
    np.random.seed(0)
    dl = delay
    np.random.seed(0)

    data = {}
    for sp in starting_points:
        lorenz = ct.lorenz_attractor(n, starting_point=sp, skip=2000)
        data[("lorenz", tuple(sp))] = lorenz
        for a in axis:
            for d in dims:
                emb = ct.embedding(lorenz[:, a].reshape((n,)), d, dl)
                # data[('emb', tuple(sp), a, d)] = emb
                data[('emb', a, d, tuple(sp))] = emb
    return data


def lorenz_homeomorphisms(data, dl=5):
    # get original lorenzes as a refference sequences
    lorenzes = {}
    for l in data:
        if l[0] == 'lorenz':
            lorenzes[l[1]] = data[l]

    def homeo(k1, k2, ts1, ts2):
        if k1[0] == 'lorenz' and k2[0] == 'emb':
            refference_sequence = ct.embedding(ts1[:, k2[2]], k2[3], dl)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == 'lorenz':
            refference_sequence = lorenzes[k1[1]]

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == 'emb':
            refference_sequence = ct.embedding(lorenzes[k1[1]][:, k2[2]], k2[3], dl)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(refference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'lorenz' and k2[0] == 'lorenz':
            return lambda x: x

    return homeo


def vanilla_experiment_lorenz(test_idx=0):
    """
    Experiment 3A described in Section 4.3.1. of https://arxiv.org/abs/2301.06753
    """
    np.random.seed(0)
    n = 10000
    # delay of an embedding
    dl = 5

    tests = ["embedding", "starting_point"]
    current_test = tests[test_idx]
    pairs = None
    if current_test == "embedding":
        starting_points = [[1., 1., 1.], [2., 1., 1.]]
        data = generate_lorenz_data_embeddings_test(n, starting_points, dl, dims=[1, 2, 3], axis=[0, 2])
        pairs = [(0, i) for i in range(0, len(data.keys()))]
        base_name = 'lorenz_emb'
    elif current_test == "starting_point":
        base_name = 'lorenz_start'
        starting_points = [[1., 1., 1.], [1.1, 1., 1.]]
        data = generate_lorenz_data_starting_test(n, starting_points, dl)

    rv = [3]
    kv = [5]
    tv = [10]

    ### for computing all vs. all set
    # pairs = None
    ct.vanilla_experiment(data, output_directory + base_name + '_n' + str(n), kv, tv, rv, True, True,
                          ct.embedding_homeomorphisms(data, 'lorenz', dl),
                          pairs=pairs, dist_fun='max', out_dir=output_directory)


def experiment_lorenz_embedding():
    """
    Experiment 3B described in Section 4.3.2. of https://arxiv.org/abs/2301.06753
    """
    base_name = output_directory + 'lorenz_dim_embedding'
    do_fnn = 1
    do_knn = 1
    do_conj = 1

    dl = 5
    tv = list(np.arange(1, 21, 4)) + list(np.arange(25, 81, 5))
    kv = list(np.arange(1, 21, 1)) + list(np.arange(25, 61, 5))
    rv = list(np.arange(1, 21, 1))
    dimsv = [1, 2, 3, 4, 5, 6]
    fnn_diffs = np.zeros((len(dimsv)-1, 2, len(rv)))
    knn_diffs = np.zeros((len(dimsv)-1, 2, len(kv)))
    neigh_conj_diffs_k = np.zeros((len(dimsv)-1, 2, len(kv)))
    neigh_conj_diffs_t = np.zeros((len(dimsv)-1, 2, len(tv)))

    data = generate_lorenz_data_embeddings_test(n=10000,
                                                starting_points=np.array([[1., 1., 1.]]),
                                                delay=dl,
                                                dims=dimsv,
                                                axis=[0])
    homeo = ct.embedding_homeomorphisms(data, 'lorenz', dl=dl)
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
            neigh_conj_diffs_t[i-1, 1, :] = ct.neigh_conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=[5], t=tv,
                                                                    dist_fun='max')
            neigh_conj_diffs_k[i-1, 0, :] = ct.neigh_conjugacy_test(tsA, tsB, homeo(k1, k2, ts1, ts2), k=kv, t=[10],
                                                                    dist_fun='max')[:, 0]
            neigh_conj_diffs_k[i-1, 1, :] = ct.neigh_conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=kv, t=[10],
                                                                    dist_fun='max')[:, 0]

    if do_fnn:
        fnn_df = pd.DataFrame(data=fnn_diffs[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(r) for r in rv])
        print('----------------------------------------------------------------------------------')
        print("FNN; column: value of r; row: dimension")
        print("time series 1 vs. time series 2")
        fnn_df.to_csv(output_directory + base_name + '_fnn_dir0' + '.csv')
        fnn_df = pd.DataFrame(data=fnn_diffs[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(r) for r in rv])
        fnn_df.to_csv(output_directory + base_name + '_fnn_dir1' + '.csv')
        print("time series 2 vs. time series 1")
        print(fnn_df.to_markdown())
    if do_knn:
        knn_df = pd.DataFrame(data=knn_diffs[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        knn_df.to_csv(output_directory + base_name + '_knn_dir0' + '.csv')
        print('----------------------------------------------------------------------------------')
        print("KNN; column: value of k; row: dimension")
        print("time series 1 vs. time series 2")
        print(knn_df.to_markdown())
        knn_df = pd.DataFrame(data=knn_diffs[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        knn_df.to_csv(output_directory + base_name + '_knn_dir1' + '.csv')
        print("time series 2 vs. time series 1")
        print(knn_df.to_markdown())
    if do_conj:
        print('----------------------------------------------------------------------------------')
        print('ConjTestPlus; column: value of k; row: dimension')
        print("time series 1 vs. time series 2")
        conj_df = pd.DataFrame(data=neigh_conj_diffs_k[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        conj_df.to_csv(output_directory + base_name + '_conj_k_dir0' + '.csv')
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_k[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(k) for k in kv])
        conj_df.to_csv(output_directory + base_name + '_conj_k_dir1' + '.csv')
        print('ConjTestPlus; column: value of k; row: dimension')
        print("time series 2 vs. time series 1")
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_t[:, 0, :], index=[str(i) for i in dimsv[:-1]], columns=[str(t) for t in tv])
        conj_df.to_csv(output_directory + base_name + '_conj_t_dir0' + '.csv')
        print('ConjTestPlus; column: value of t; row: dimension')
        print("time series 1 vs. time series 2")
        print(conj_df.to_markdown())
        conj_df = pd.DataFrame(data=neigh_conj_diffs_t[:, 1, :], index=[str(i) for i in dimsv[:-1]], columns=[str(t) for t in tv])
        conj_df.to_csv(output_directory + base_name + '_conj_t_dir1' + '.csv')
        print('ConjTestPlus; column: value of t; row: dimension')
        print("time series 2 vs. time series 1")
        print(conj_df.to_markdown())


def experiment_lorenz_diff_sp_grid():
    """
    Experiment 3C described in Section 4.3.3. of https://arxiv.org/abs/2301.06753
    """
    dl = 5
    for embv in [1, 2, 3, 4]:
        data = generate_lorenz_data_starting_test(n=10000,
                                                  starting_points=np.array([[1., 1., 1.], [1., 2., 1.], [2., 1., 1.], [1., 1., 2.]]),
                                                  delay=dl,
                                                  emb_dim=embv)
        kv = [5]
        tv = list(np.arange(1, 12, 2)) + list(np.arange(15, 26, 5)) + list(np.arange(30, 101, 10))
        keys = [k for k in data.keys()]

        conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))
        neigh_conj_diffs = np.zeros((len(keys) - 1, len(kv), len(tv)))

        homeo = ct.embedding_homeomorphisms(data, 'lorenz', dl=dl)

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
                    base_name = output_directory + '/lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[2]) + str(k1[3]) + '_vs_' + k2[0] + str(k2[1])
                elif embv == 2:
                    base_name = output_directory + '/lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
                else:
                    base_name = 'wat?'
                    continue
            else:
                k2 = keys[j]
                ts2 = data[k2]
                if k2[0] == 'emb':
                    base_name = output_directory + '/lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[3]) + str(k2[1])
                else:
                    base_name = output_directory + '/lorenz_stpts_grid_dmax_t_' + k1[0] + str(k1[1]) + '_vs_' + k2[0] + str(k2[1])
            print(k1, k2)
            new_n = min(len(ts1), len(ts2))

            print(base_name)
            conj_diffs[j - 1, :, :] = ct.conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv, t=tv,
                                                        dist_fun='max')
            conj_df = pd.DataFrame(data=conj_diffs[j - 1, :, :], index=[str(k) for k in kv], columns=[str(t) for t in tv])
            conj_df.to_csv(output_directory + '/' + base_name + '_conj.csv')
            print(conj_df.to_markdown())

            neigh_conj_diffs[j - 1, :, :] = ct.neigh_conjugacy_test(ts1[:new_n], ts2[:new_n], homeo(k1, k2, ts1, ts2), k=kv,
                                                                    t=tv, dist_fun='max')
            neigh_conj_df = pd.DataFrame(data=neigh_conj_diffs[j - 1, :, :], index=[str(k) for k in kv],
                                         columns=[str(t) for t in tv])
            neigh_conj_df.to_csv(output_directory + '/' + base_name + '_neigh_conj.csv')
            print(neigh_conj_df.to_markdown())


if __name__ == '__main__':
    os.makedirs(output_directory, exist_ok=True)

    ### experiment 3A - 4.3.1
    vanilla_experiment_lorenz(0)

    ### experiment 3B - 4.3.2
    experiment_lorenz_embedding()

    ### experiment 3C - 4.3.3
    experiment_lorenz_diff_sp_grid()
