import json
import uuid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm, linear_model, neighbors
from sklearn.decomposition import PCA
from pprint import pprint as pp
from scipy.ndimage import gaussian_filter1d as blur



def main():
    # path = "processed/color_feature3_nofilter.npy"
    stradegy = "dominant_color_vec_feature"
    v = 2
    query = 'plate'

    path = "processed/{}{}_{}_{}.npy".format(stradegy, v, 'mic', query)
    # path =
    data = np.load(path)
    outp = "processed/{}{}_{}_{}.npy".format(stradegy, v, 'neg', query)
    # outp = "processed/dominant_color_vec_feature{}_neg_plate.npy".format(v)
    outliers = np.load(outp)
    # outliers = np.load("random_colors_processed.npy")
    sigma = 1
    # print("gamma=\tTrain Data\tTest Data")
    # for sigma in [s/100 for s in range(1,300,3)]:
    # print("{}".format(None), end="\t")
    #     fdata = blur(data, sigma=sigma, axis=1, mode='constant')
    #     foutliers = blur(outliers, sigma=sigma, axis=1, mode='constant')
    # data = outliers
    # for i in range(20):
    #     plt.plot(data[i], 'k', label='original data')
    #     plt.plot(blur(data[i], 3), '--', label='filtered, sigma=3')
    #     plt.plot(blur(data[i], 6), ':', label='filtered, sigma=6')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    # visual(data, outliers, save=True)
    learn_svc(data, outliers)
    learn_uniclass(data, outliers)
    learn_SGD(data, outliers)
    learn_neighbors(data, outliers)

def build_penalty_vec(src, query, i):
    with open(src, 'r') as f:
        jindex = json.load(f)

    query_index = jindex["queries"][i]
    assert query_index["query"] ==  query
    relevant = sorted(query_index["images"])
    vec = np.zeros((len(relevant)))
    # for score_obj in query_index["scores"]

def flatten(data, new_shape):
    return data.reshape(new_shape)

def visual(data, outliers=None, save=None):
    pca = PCA(n_components=2)
    leng = 1
    for m in data.shape[1:]:
        leng *= m
    flat = flatten(data, (data.shape[0], leng))

    res = pca.fit_transform(flat)
    xs = res[..., 0]
    ys = res[..., 1]
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = plt.gca()
    plt.scatter(xs, ys, c='navy', marker="P", label="Michelin Dishes")
    if outliers is not None:
        flat_o = flatten(outliers, (outliers.shape[0], leng))
        # flat_o = outliers.reshape(outliers.shape[0], outliers.shape[1] * outliers.shape[2] * outliers.shape[3])

        outs = pca.transform(flat_o)
        oxs = outs[..., 0]
        oys = outs[..., 1]
        plt.scatter(oxs, oys, c='brown', marker="v", label="Foodphotography Dishes")
    plt.legend()
    plt.show()
    # if save:
    #     plt.savefig("plots/{}".format(uuid.uuid4()))

def learn_SGD(data, outliers=None):
    np.random.shuffle(data)
    leng = 1
    for m in data.shape[1:]:
        leng *= m
    flat = flatten(data, (data.shape[0], leng))

    if outliers is None:
        outliers = np.random.uniform(low=0, high=1, size=(data.shape))
    np.random.shuffle(outliers)
    flat_o = flatten(outliers, (outliers.shape[0], leng))
    split_1 = int(0.9*data.shape[0])
    split_neg = int(0.9*outliers.shape[0])
    train = np.concatenate((flat[:split_1], flat_o[:split_neg]))
    test = np.concatenate((flat[split_1:], flat_o[split_neg:]))
    labels = [1] * split_1 + [-1] * split_neg
    expected = [1] * (data.shape[0] - split_1) + [-1] * (outliers.shape[0] - split_neg)
    max_iter = 1000
    clf = linear_model.SGDClassifier(max_iter=max_iter)
    clf.fit(train, labels)
    y_pred_train = clf.predict(train)
    y_pred_test = clf.predict(test)
    y_pred_outliers = clf.predict(flat_o)
    label_err = y_pred_train != labels
    test_err = y_pred_test != expected
    print("SGD C:\tmax_iter={}\tl_err={}\tt_err={}\tl_g={}\tl_b={}\tt_g={}\tt_b={}\t".format(
        max_iter,
        round(sum(label_err) / len(labels), 2),
        round(sum(test_err) / len(expected), 2),
        round(sum(label_err[:split_1]) / len(label_err), 2),
        round(sum(label_err[split_1:]) / len(label_err), 2),
        round(sum(test_err[:(data.shape[0] - split_1)])/len(test_err), 2),
        round(sum(test_err[(data.shape[0] - split_1):])/len(test_err), 2))
          )
    assert len(expected) == len(test_err)


def learn_neighbors(data, outliers=None):
    np.random.shuffle(data)
    leng = 1
    for m in data.shape[1:]:
        leng *= m
    flat = flatten(data, (data.shape[0], leng))

    if outliers is None:
        outliers = np.random.uniform(low=0, high=1, size=(data.shape))
    np.random.shuffle(outliers)
    flat_o = flatten(outliers, (outliers.shape[0], leng))
    split_1 = int(0.8*data.shape[0])
    split_neg = int(0.8*outliers.shape[0])
    train = np.concatenate((flat[:split_1], flat_o[:split_neg]))
    test = np.concatenate((flat[split_1:], flat_o[split_neg:]))
    labels = [1] * split_1 + [-1] * split_neg
    expected = [1] * (data.shape[0] - split_1) + [-1] * (outliers.shape[0] - split_neg)

    assert train.shape[0] == labels.__len__()

    gamma = 'auto'
    for n_neighbs in range(3, 10, 1):
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbs)
        clf.fit(train, labels)
        y_pred_train = clf.predict(train)
        y_pred_test = clf.predict(test)
        y_pred_outliers = clf.predict(flat_o)
        label_err = y_pred_train != labels
        test_err = y_pred_test != expected
        print("nearest neighbors:\tgamma={}\tn_neighb={}\tl_err={}\tt_err={}\tlg={}\tlb={}\ttg={}\ttb={}\t".format(
            gamma,
            n_neighbs,
            round(sum(label_err) / len(labels), 2),
            round(sum(test_err) / len(expected), 2),
            round(sum(label_err[:split_1]) / len(label_err), 2),
            round(sum(label_err[split_1:]) / len(label_err), 2),
            round(sum(test_err[:(data.shape[0] - split_1)])/len(test_err), 2),
            round(sum(test_err[(data.shape[0] - split_1):])/len(test_err), 2))
              )
        assert len(expected) == len(test_err)


def learn_uniclass(data, outliers=None):
    s = data.shape[0]
    np.random.shuffle(data)
    leng = 1
    for m in data.shape[1:]:
        leng *= m
    flat = flatten(data, (data.shape[0], leng))
    split = int(0.8*s)
    train = flat[:split]
    test = flat[split:]
    # print("max value in data :", np.max(data))
    # print("min value in data :", np.min(data))
    if outliers is None:
        outliers = np.random.uniform(low=0, high=1, size=(data.shape))
    flat_o = flatten(outliers, (outliers.shape[0], leng))
    # print("max value in outliers :", np.max(outliers))
    # print("min value in outliers :", np.min(outliers))
    # flat_o = outliers.reshape(outliers.shape[0], outliers.shape[1] * outliers.shape[2] * outliers.shape[3])
    # print("gamma=\tnu=\tFalse Negatives\tFalse Positive")
    # for nu in range(1, 100, 100):
    for gamma in ['auto']:
        for nu in [0.2, 0.4, 0.6, 0.8]:
            # for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
            # print ("nu: " + str(nu))
            clf = svm.OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
            clf.fit(train)
            y_pred_train = clf.predict(train)
            y_pred_test = clf.predict(test)
            y_pred_outliers = clf.predict(flat_o)
            n = y_pred_test == -1
            m = y_pred_outliers == 1

            print("OneClassSVM\tgamma={}\tnu={}\tfn={}%\tfp={}%".format(gamma, nu,
                                                                        round(sum(n)/(sum(n)+sum(~n)), 2) * 100,
                                                                        round(sum(m)/(sum(m)+sum(~m)), 2) * 100))

    # print(test.size)
    # print(sum(y_pred_test))

def learn_svc(data, outliers=None):
    np.random.shuffle(data)
    leng = 1
    for m in data.shape[1:]:
        leng *= m
    flat = flatten(data, (data.shape[0], leng))

    if outliers is None:
        outliers = np.random.uniform(low=0, high=1, size=(data.shape))
    np.random.shuffle(outliers)
    flat_o = flatten(outliers, (outliers.shape[0], leng))
    split_1 = int(0.8*data.shape[0])
    split_neg = int(0.8*outliers.shape[0])
    train = np.concatenate((flat[:split_1], flat_o[:split_neg]))
    test = np.concatenate((flat[split_1:], flat_o[split_neg:]))
    labels = [1] * split_1 + [-1] * split_neg
    expected = [1] * (data.shape[0] - split_1) + [-1] * (outliers.shape[0] - split_neg)

    # print("max value in data :", np.max(data))
    # print("min value in data :", np.min(data))
    #
    # print("max value in outliers :", np.max(outliers))
    # print("min value in outliers :", np.min(outliers))
    # flat_o = outliers.reshape(outliers.shape[0], outliers.shape[1] * outliers.shape[2] * outliers.shape[3])
    # print("gamma=\tTrain Data\tTest Data")
    # for nu in range(1, 100, 100):
    gamma = 'auto'
    for nu in [s/1000 for s in range(3, 75, 15)]:
    # for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
        # print ("nu: " + str(nu))
        # clf = svm.LinearSVC()
        clf = svm.NuSVC(nu=nu, kernel='rbf', gamma=gamma)
        clf.fit(train, labels)
        y_pred_train = clf.predict(train)
        y_pred_test = clf.predict(test)
        y_pred_outliers = clf.predict(flat_o)
        label_err = y_pred_train != labels
        test_err = y_pred_test != expected
        print("NuSVC:\tgamma={}\tnu={}\tl_err={}\tt_err={}\tlg={}\tlb={}\ttg={}\ttb={}\t".format(
            gamma,
            nu,
            round(sum(label_err) / len(labels), 2),
            round(sum(test_err) / len(expected), 2),
            round(sum(label_err[:split_1]) / len(label_err), 2),
            round(sum(label_err[split_1:]) / len(label_err), 2),
            round(sum(test_err[:(data.shape[0] - split_1)])/len(test_err), 2),
            round(sum(test_err[(data.shape[0] - split_1):])/len(test_err), 2))
              )
        assert len(expected) == len(test_err)

        # print( "\tFalse Positive:")
        # print(sum(m), " / ", (sum(m)+sum(~m)))
        # print(round(sum(m)/(sum(m)+sum(~m)), 2) * 100, " %")
        # print()
    # print(test.size)
    # print(sum(y_pred_test))




def other():
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate train data
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                     edgecolors='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                    edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/200 ; errors novel regular: %d/40 ; "
        "errors novel abnormal: %d/40"
        % (n_error_train, n_error_test, n_error_outliers))
    plt.show()

main()