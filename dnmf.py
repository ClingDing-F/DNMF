import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold


def modeluar_p(X):
    return (abs(X) + X) / 2


def modeluar_n(X):
    return (abs(X) - X) / 2


class DNMF(object):
    def __init__(self, rank=2, max_iters=2000, alpha=0.5, eps=1e-5, output=False, seed=0):

        self.rank = rank
        self.U_matrix = []
        self.V_train = []
        self.A_matrix = []
        self.V_test = []

        self.Objective_error_train = []
        self.Objective_error_test = []
        self.max_iters_train = max_iters
        self.max_iters_test = int(max_iters / 1.5)
        self.alpha = alpha
        self.output = output
        self.eps = eps
        if seed != 0:
            self.seed = np.random.seed(seed)
        else:
            self.seed = seed

    def fit(self, Data_matrix, labels, classes, label_ratio, U_matrix=False):

        Data_matrix = Data_matrix.T
        labels = labels
        Classes = classes
        Data_num = len(labels)
        label_matrix = np.zeros((Data_num, classes))
        for i in range(Data_num):
            label_matrix[i][labels[i]] = 1
        Q = label_matrix.T

        label_matrix = label_matrix
        M, N = Data_matrix.shape
        flag = True
        if isinstance(U_matrix, np.ndarray):
            U = U_matrix
            flag = False
        else:
            U = np.random.rand(M, self.rank)
        V = np.random.rand(N, self.rank)
        V_l = V.copy()
        V_l[int(N * label_ratio):] = 0
        if self.output:
            print("Data Matrix shape m*n: {}".format(Data_matrix.shape))
            print("Label Matrix shape m*n: {}".format(label_matrix.shape))
            print("Classes: \t {} label_ratio: \t {}".format(Classes, label_ratio))
            print("U Matrix shape m*rank: {}".format(U.shape))
            print("V Matrix shape rank*n: {}".format(V.shape))
            print("Q Matrix shape classes* rank: {}".format(Q.shape))
        # loss function:
        A = np.random.randn(Classes, self.rank)

        X = Data_matrix
        self.Objective_error_train = []
        L = np.power(np.linalg.norm(X - U @ V.T), 2) + self.alpha * np.power(np.linalg.norm(Q - A @ V.T), 2)
        self.Objective_error_train.append(L)
        # train
        for iters in range(self.max_iters_train):
            V_l = V.copy()
            V_l[int(N * label_ratio):] = 0
            if flag:
                U *= np.dot(X, V) / np.dot(np.dot(U, V.T), V)

            V *= ((np.dot(X.T, U) + self.alpha * (modeluar_n(np.dot(np.dot(V_l, A.T), A))) + self.alpha * modeluar_p(
                np.dot(Q.T, A))) /
                  (np.dot(np.dot(V, U.T), U) + self.alpha * (
                      modeluar_p(np.dot(np.dot(V_l, A.T), A))) + self.alpha * modeluar_n(np.dot(Q.T, A))))

            A = np.dot(np.dot(Q, V_l), np.linalg.pinv(np.dot(V_l.T, V_l)))

            U = np.where(U > self.eps, U, 0)
            V = np.where(V > self.eps, V, 0)

            L = np.power(np.linalg.norm(X - U @ V.T), 2) + self.alpha * np.power(np.linalg.norm(Q - A @ V_l.T), 2)
            self.Objective_error_train.append(L)

        self.U_matrix = U
        self.V_train = V
        self.A_matrix = A
        return self

    def transform(self, Data_matrix):

        X = Data_matrix.T
        M, N = X.shape
        V = np.random.rand(N, self.rank)
        U = self.U_matrix
        self.Objective_error_test = []
        for i in range(self.max_iters_test):
            V *= np.dot(X.T, U) / np.dot(np.dot(V, U.T), U)
            L = np.power(np.linalg.norm(X - U @ V.T), 2)
            self.Objective_error_test.append(L)

        self.V_test = V

        return V


def KNN_results(X_train, Y_train, X_test, Y_test, neighbours=10):
    knn3 = KNeighborsClassifier(n_neighbors=neighbours, weights="distance", metric='euclidean')
    knn3.fit(X_train, Y_train)
    score = knn3.score(X_test, Y_test)
    return score


def Draw_KFold(X,Y,fold=5,rank=2, max_iters=2000, alpha=0.5, output=False, seed=0):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 1
    fig = plt.figure(figsize=(5, 5))
    sns.set(style="whitegrid", font_scale=1.2)
    KF = KFold(n_splits=fold, shuffle=True, random_state=1)
    # KF = KFold(n_splits=3,shuffle=False)
    for train_index, test_index in KF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        #     ORi
        dnmf = DNMF(rank, max_iters, alpha, output, seed)
        dnmf.fit(Data_matrix=X_train, labels=Y_train, classes=2, U_matrix=False)
        V_test = dnmf.transform(Data_matrix=X_test)

        knn1 = KNeighborsClassifier(n_neighbors=10, weights="distance", metric='euclidean')
        knn1.fit(dnmf.V_train, Y_train)
        Ori_y_scores = knn1.predict_proba(V_test)

        fpr, tpr, threshold = roc_curve(Y_test, Ori_y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, lw=2, alpha=0.5, label='Fold %d (AUC=%0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean:(AUC=%0.2f$\pm$%0.2f)' % (mean_auc, std_auc), lw=3, alpha=1)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.legend(loc=4, prop={'size': 10, 'weight': 'bold'})
    plt.show()
