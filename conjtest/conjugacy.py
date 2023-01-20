import numba
import numpy as np
from scipy.spatial.distance import cdist, directed_hausdorff
from hausdorff import hausdorff_distance  ### https://github.com/mavillan/py-hausdorff


@numba.jit(nopython=True, fastmath=True)
def max_dist(x, y):
    return max(np.abs(x - y))


def conjugacy_test_knn(ts1, ts2, k=None, point_vals=False, dist_fun=None):
    """
    @param ts1:
    @param ts2:
    @param k: a list of parameters k for which values should be computed
    @param point_vals:
    @param dist_fun:
    @return:
    """
    if k is None:
        k = [1]
    assert len(ts1) == len(ts2)
    n = len(ts1)
    if dist_fun is None:
        distf = 'euclidean'
    elif dist_fun == 'max':
        distf = 'chebyshev'
    else:
        distf = dist_fun

    # distances between points within time series
    dists1 = cdist(ts1, ts1, distf)
    dists2 = cdist(ts2, ts2, distf)
    # sorted indexes for every point according to the distance
    nn1 = np.argsort(dists1, axis=1)
    nn2 = np.argsort(dists2, axis=1)

    knns_first_vs_second = []
    local_diffs_1 = []
    knns_second_vs_first = []
    local_diffs_2 = []
    for kv in k:
        diff12 = 0
        pvals12 = []
        pvals21 = []
        # compare ts1 to ts2
        for i in range(n):
            # take indices of k-nearest neighbours of x_i (kv+1 because it has distance 0 to itself)
            knn1 = set(nn1[i, :kv + 1])
            diff = -len(knn1)
            j = 0
            # from the list of k-nn remove nearest neighbours of y_i until the list knn1 is empty
            # if the neighbours are the same diff would be 0 at the end
            while len(knn1) > 0:
                knn1 = knn1.difference([nn2[i, j]])
                diff += 1
                j += 1
            pvals12.append(diff)
            diff12 += diff
        # divide by n^2 - we want to measure how relatively many artificial neighbours are created,
        # the second n comes from computing the average value over all points
        diff12 = diff12 / (n * n)
        knns_first_vs_second.append(diff12)
        local_diffs_1.append(pvals12)

    for kv in k:
        diff21 = 0
        # compare ts2 to ts1
        for i in range(n):
            knn2 = set(nn2[i, :kv + 1])
            diff = -len(knn2)
            j = 0
            while len(knn2) > 0:
                knn2 = knn2.difference([nn1[i, j]])
                diff += 1
                j += 1
            diff21 += diff
            pvals21.append(diff)
        diff21 = diff21 / (n * n)
        knns_second_vs_first.append(diff21)
        local_diffs_2.append(pvals21)

    if point_vals:
        return knns_first_vs_second, knns_second_vs_first, local_diffs_1, local_diffs_2
    else:
        return knns_first_vs_second, knns_second_vs_first


def symmetric_conjugacy_knn(ts1, ts2, k, dist_fun=None):
    """
    The assumption the conjugacy h:X->Y maps h(x_i) = y_i
    :param ts1: time series one - an array of size [n, d]
    :param ts2: time series two - shape the same as ts1
    :param k: the number of closest neigbors to be taken into account
    :param dist_fun: metric to be used
    :return:
    """
    assert len(ts1) == len(ts2)
    n = len(ts1)
    if dist_fun is None:
        distf = 'euclidean'
    elif dist_fun == 'max':
        distf = 'chebyshev'
    else:
        distf = dist_fun

    dists1 = cdist(ts1, ts1, distf)
    dists2 = cdist(ts2, ts2, distf)
    nn1 = np.argsort(dists1, axis=1)
    nn2 = np.argsort(dists2, axis=1)

    total_diff = 0
    for i in range(n):
        knn1 = set(nn1[i, :k + 1])
        diff1 = -len(knn1)
        j = 0
        while len(knn1) > 0:
            knn1 = knn1.difference([nn2[i, j]])
            diff1 += 1
            j += 1

        knn2 = set(nn2[i, :k + 1])
        diff2 = -len(knn2)
        j = 0
        while len(knn2) > 0:
            knn2 = knn2.difference([nn1[i, j]])
            diff2 += 1
            j += 1
        total_diff = max(diff1, diff2)

    return total_diff / n


def fnn(ts1, ts2, r=None, dist_fun=None):
    """
    :param ts1: time series one - an array of size [n, d]
    :param ts2: time series two - shape the same as ts1
    :param r: the parameter of FNN method
    :param dist_fun: metric to be used
    :return:
    """
    if dist_fun is None:
        distf = 'euclidean'
    elif dist_fun == 'max':
        distf = 'chebyshev'
    else:
        distf = dist_fun
    if r is None:
        r = [2]
    assert len(ts1) == len(ts2)
    n = len(ts1)

    # compute distances within a time series
    dists1 = cdist(ts1, ts1, distf)
    dists2 = cdist(ts2, ts2, distf)
    # make the distance to itself large
    dists1 = dists1 + np.diag(np.ones((n,)) * 2 * np.max(dists1))
    dists2 = dists2 + np.diag(np.ones((n,)) * 2 * np.max(dists2))
    # compute standard deviation among all distances (we take only the upper triangular entries)
    std1 = np.std(dists1[np.triu_indices(len(dists1), k=1)])
    std2 = np.std(dists2[np.triu_indices(len(dists2), k=1)])

    # nearest neighbors
    nn1 = np.argmin(dists1, axis=1)
    nn2 = np.argmin(dists2, axis=1)

    def H(x):
        return 1 if x > 0 else 0

    fnns1 = []
    fnns2 = []
    # TODO: it would me more efficient to inverse the loops
    for rv in r:
        fnn1_div = 0
        fnn2_div = 0
        fnn1_num = 0
        fnn2_num = 0
        for i in range(n):
            v1 = H(std1 / rv - dists1[i, nn1[i]])
            fnn1_num += H(dists2[i, nn1[i]] / dists1[i, nn1[i]] - rv) * v1
            fnn1_div += v1

            v2 = H(std2 / rv - dists2[i, nn2[i]])
            fnn2_num += H(dists1[i, nn2[i]] / dists2[i, nn2[i]] - rv) * v2
            fnn2_div += v2

        if fnn1_div == 0:
            fnns1.append(np.infty)
        else:
            fnns1.append(fnn1_num / fnn1_div)
        if fnn2_div == 0:
            fnns2.append(np.infty)
        else:
            fnns2.append(fnn2_num / fnn2_div)

    return fnns1, fnns2


def conjugacy_test(tsX, tsY, h, k=None, t=None, dist_fun=None):
    """
    Conjugacy that does not require the direct correspondence of time series
    :param tsX: a time series in X
    :param tsY: a time series in Y
    :param h: a map from X to Y
    :param k: k-nn are taken as an approximation of a neighbourhood, default=1
    :param t: conjugacy is tested t steps forward, default=1
    :return:
    """

    if k is None:
        k = [1]
    if t is None:
        t = [1]

    if dist_fun is None:
        distf = 'euclidean'
    elif dist_fun == 'max':
        distf = 'chebyshev'
    else:
        distf = dist_fun

    if h is None:
        return np.ones((len(k), len(t))) * np.infty
    maxk = np.max(k)
    maxt = np.max(t)

    distsX = cdist(tsX[:-maxt], tsX[:-maxt], distf)

    nnX = np.argsort(distsX, axis=1)
    # nnY = np.argsort(distsY, axis=1)

    accumulated_hausdorff = {}
    for tv in t:
        for kv in k:
            accumulated_hausdorff[(kv, tv)] = []

    # mk in the variable name denotes that someting is for maxk, only 'k' denotes that it is k-specific
    for i in range(len(tsX) - maxt):
        # take h images of k nearest neigh. of x_i - h(Ux)
        hmknnX = h(np.array([tsX[x] for x in nnX[i, :maxk + 1]]))
        # consider a special case when hknnX is a singleton in 1-D
        if len(hmknnX.shape) == 1:
            hmknnX = hmknnX.reshape((len(hmknnX), 1))
        elif hmknnX.shape[1] == 1 and hmknnX.shape[0] == 1:
            hmknnX = hmknnX.reshape((1, 1))
        # print(hmknnX.shape, tsY[:-maxt].shape)
        # take h images of k nearest neigh. of x_i - h(Ux) and compute distances to points in Y
        knns_dists = cdist(hmknnX, tsY[:-maxt], distf)

        for it, tv in enumerate(t):
            # push t-times k-neigh of x_i forward and then compute the image - h(f^t(Ux))
            hfmknnX = h(np.array([tsX[x + tv] for x in nnX[i, :maxk + 1]]))
            # find indices of nearest neighbours of points in that image - denote it Vx
            # TODO: is this maxk here unnecessary?
            idx_mknnY = np.argmin(knns_dists[:maxk + 1], axis=1)
            for ik, kv in enumerate(k):
                # take indices of k nearest neigh. of x_i - denote it Ux
                # idx_knnX = nnX[i, :kv + 1]
                # push t-times k-neigh of x_i forward and then compute the image - h(f^t(Ux))
                hfknnX = hfmknnX[:kv + 1]
                if len(hfknnX.shape) == 1:
                    hfknnX = hfknnX.reshape((len(hfknnX), 1))

                idx_knnY = idx_mknnY[:kv + 1]
                # push t-times the h image k-neigh of x_i forward - g^t(Vx)
                gknnY = np.array([tsY[y + tv] for y in idx_knnY])

                im_hdist = hausdorff_distance(hfknnX, gknnY, distf)
                accumulated_hausdorff[(kv, tv)].append(im_hdist)

    distsY = cdist(tsY, tsY, distf)
    max_distY = np.max(distsY)
    diffs = np.zeros((len(k), len(t)))
    for it, tv in enumerate(t):
        for ik, kv in enumerate(k):
            diffs[ik, it] = np.sum(accumulated_hausdorff[(kv, tv)]) / (len(accumulated_hausdorff[(kv, tv)]) * max_distY)

    # print(diffs)
    return diffs


def neigh_conjugacy_test(tsX, tsY, h, k=None, t=None, dist_fun=None):
    """
    Conjugacy that does not require the direct correspondence of time series
    :param tsX: a time series in X
    :param tsY: a time series in Y
    :param h: a map from X to Y
    :param k: k-nn are taken as an approximation of a neighbourhood, default=1
    :param t: conjugacy is tested t steps forward, default=1
    :return:
    """

    if k is None:
        k = [1]
    if t is None:
        t = [1]

    if dist_fun is None:
        distf = 'euclidean'
    elif dist_fun == 'max':
        distf = 'chebyshev'
    else:
        distf = dist_fun

    if h is None:
        return np.ones((len(k), len(t))) * np.infty
    maxk = np.max(k)
    maxt = np.max(t)

    distsX = cdist(tsX[:-maxt], tsX[:-2* maxt], distf)
    distsY = cdist(tsY[:-maxt], tsY[:-2* maxt], distf)

    nnX = np.argsort(distsX, axis=1)
    nnY = np.argsort(distsY, axis=1)

    accumulated_hausdorff = {}
    for tv in t:
        for kv in k:
            accumulated_hausdorff[(kv, tv)] = []
    # mk in the variable name denotes that someting is for maxk, only 'k' denotes that it is k-specific
    for i in range(len(tsX) - maxt - 1):
        # take h images of k nearest neigh. of x_i - h(Ux)
        hmknnX = h(np.array([tsX[x] for x in nnX[i, :maxk + 1]]))
        # consider a special case when hknnX is a singleton in 1-D
        if len(hmknnX.shape) == 1:
            hmknnX = hmknnX.reshape((len(hmknnX), 1))
        elif hmknnX.shape[1] == 1 and hmknnX.shape[0] == 1:
            hmknnX = hmknnX.reshape((1, 1))
        # take h images of k nearest neigh. of x_i - h(Ux) and compute distances to points in Y
        knns_dists = cdist(hmknnX, tsY[:-2*maxt], distf)

        for it, tv in enumerate(t):
            # push t-times k-neigh of x_i forward and then compute the image - h(f^t(Ux))
            hfmknnX = h(np.array([tsX[x + tv] for x in nnX[i, :maxk + 1]]))
            # find indices of nearest neighbours of points in that image - denote it Vx
            idx_mknnY = np.argmin(knns_dists, axis=1)
            for ik, kv in enumerate(k):
                # push t-times k-neigh of x_i forward and then compute the image - h(f^t(Ux))
                hfknnX = hfmknnX[:kv + 1]
                if len(hfknnX.shape) == 1:
                    hfknnX = hfknnX.reshape((len(hfknnX), 1))

                # find the minimal neighborhood of y containing all kv of hknnY
                max_hknnY_idx = (nnY[idx_mknnY[0], :, None] == idx_mknnY[:kv + 1]).argmax(axis=0).max() + 1
                idx_knnY = nnY[idx_mknnY[0], :max_hknnY_idx]
                # push t-times the h image k-neigh of x_i forward - g^t(Vx)
                ghknnY = np.array([tsY[y + tv] for y in idx_knnY])

                im_hdist = hausdorff_distance(hfknnX, ghknnY, distf)
                accumulated_hausdorff[(kv, tv)].append(im_hdist)

    distsY = cdist(tsY, tsY, distf)
    max_distY = np.max(distsY)
    diffs = np.zeros((len(k), len(t)))
    for it, tv in enumerate(t):
        for ik, kv in enumerate(k):
            diffs[ik, it] = np.sum(accumulated_hausdorff[(kv, tv)]) / (len(accumulated_hausdorff[(kv, tv)]) * max_distY)

    return diffs
