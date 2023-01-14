from .conjugacy import conjugacy_test, conjugacy_test_knn, neigh_conjugacy_test, fnn
from itertools import combinations
import numpy as np
import pandas as pd


def embedding(points, dimension, delay):
    """

    :param points:
    :param dimension:
    :param delay:
    :return:
    """
    emb_indices = np.arange(dimension) * (delay + 1) + np.arange(
        np.max(points.shape[0] - (dimension - 1) * (delay + 1), 0)).reshape(-1, 1)
    ep = points[emb_indices]
    if len(points.shape) == 1:
        return ep
    else:
        return ep[:, :, 0]


def embedding_homeomorphisms(data, base_ts, dl=5):
    """

    :param data:
    :param base_ts:
    :param dl:
    :return:
    """
    # the structure of the base time series:
    # (base_ts, [parameter of a time series], [parameter2], ...)

    # get original time series serving later as a refference sequences
    base_time_series = {}
    for l in data:
        if l[0] == base_ts:
            base_time_series[l[1:]] = data[l]

    def homeo(k1, k2, ts1, ts2):
        if k1[0] == base_ts and k2[0] == 'emb':
            reference_sequence = embedding(ts1[:, k2[1]], k2[2], dl)
            # print(refference_sequence.shape)

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(reference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == base_ts:
            reference_sequence = base_time_series[k1[3:]]

            def h(x):
                points = []
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(reference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == 'emb' and k2[0] == 'emb':
            reference_sequence = embedding(base_time_series[k1[3:]][:, k2[1]], k2[2], dl)

            def h(x):
                points = []
                # print(x)
                for p in x:
                    idx = np.argwhere(ts1 == p)[0, 0]
                    points.append(reference_sequence[idx])
                return np.array(points)

            return h
        if k1[0] == base_ts and k2[0] == base_ts:
            return id

    return homeo


def vanilla_experiment(data, base_name, kv, tv, rv, do_knn, do_conj, homeo=None, pairs=None, dist_fun=None, out_dir=''):
    """

    :param data:
    :param base_name:
    :param kv:
    :param tv:
    :param rv:
    :param do_knn:
    :param do_conj:
    :param homeo:
    :param pairs:
    :param dist_fun:
    :param out_dir:
    """
    keys = [k for k in data.keys()]
    labels = [str(k) for k in keys]
    print(data.keys())

    knn_diffs = np.ones((len(data), len(data), len(kv))) * np.infty
    fnn_diffs = np.ones((len(data), len(data), len(rv))) * np.infty
    conj_diffs = np.ones((len(data), len(data), len(kv), len(tv))) * np.infty
    neigh_conj_diffs = np.ones((len(data), len(data), len(kv), len(tv))) * np.infty

    if pairs is None:
        pairs = combinations(range(len(data)), 2)

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
        if do_knn:
            knn1, knn2 = conjugacy_test_knn(ts1[:new_n], ts2[:new_n], k=kv, dist_fun=dist_fun)
            knn_diffs[i, j, :] = knn1
            knn_diffs[j, i, :] = knn2

            fnn1, fnn2 = fnn(ts1[:new_n], ts2[:new_n], r=rv, dist_fun=dist_fun)
            fnn_diffs[i, j, :] = fnn1
            fnn_diffs[j, i, :] = fnn2

        # if do_conj and k1[1] != k2[1]:
        if do_conj:
            tsA = ts1[:new_n]
            tsB = ts2[:new_n]
            conj_diffs[i, j, :, :] = conjugacy_test(tsA, tsB, homeo(k1, k2, ts1, ts2), k=kv, t=tv, dist_fun=dist_fun)
            conj_diffs[j, i, :, :] = conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=kv, t=tv, dist_fun=dist_fun)
            neigh_conj_diffs[i, j, :, :] = neigh_conjugacy_test(tsA, tsB, homeo(k1, k2, ts1, ts2), k=kv, t=tv,
                                                                   dist_fun=dist_fun)
            neigh_conj_diffs[j, i, :, :] = neigh_conjugacy_test(tsB, tsA, homeo(k2, k1, ts2, ts1), k=kv, t=tv,
                                                                   dist_fun=dist_fun)

    if do_knn:
        for ik, k in enumerate(kv):
            knn_df = pd.DataFrame(data=knn_diffs[:, :, ik], index=labels, columns=labels)
            knn_df.to_csv(out_dir + '/' + base_name + '_knns_k' + str(k) + '.csv')
            print('----------------------------------------------------------------------------------')
            print("KNN - k: " + str(k))
            print(knn_df.to_markdown())

        for ir, r in enumerate(rv):
            fnn_df = pd.DataFrame(data=fnn_diffs[:, :, ir], index=labels, columns=labels)
            fnn_df.to_csv(out_dir + '/' + base_name + '_fnns_r' + str(r) + '.csv')
            print('----------------------------------------------------------------------------------')
            print("FNN - r: " + str(r))
            print(fnn_df.to_markdown())

    if do_conj:
        for ik, k in enumerate(kv):
            for it, t in enumerate(tv):
                conj_df = pd.DataFrame(data=conj_diffs[:, :, ik, it], index=labels, columns=labels)
                conj_df.to_csv(out_dir + '/' + base_name + '_conjtest_k' + str(k) + '_t' + str(t) + '.csv')
                print('----------------------------------------------------------------------------------')
                print('ConjTest - k: ' + str(k) + ', t: ' + str(t))
                print(conj_df.to_markdown())
                neigh_conj_df = pd.DataFrame(data=neigh_conj_diffs[:, :, ik, it], index=labels, columns=labels)
                neigh_conj_df.to_csv(out_dir + '/' + base_name + '_conjtestplus_k' + str(k) + '_t' + str(t) + '.csv')
                print('----------------------------------------------------------------------------------')
                print('ConjTestPlus - k: ' + str(k) + ', t: ' + str(t))
                print(neigh_conj_df.to_markdown())
