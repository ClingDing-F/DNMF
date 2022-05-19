import dnmf as dnmf
from utils import generate_2_class_data
from sklearn.neighbors import KNeighborsClassifier



if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = generate_2_class_data(data_num=135, dim=512, bias=0.2)

    dnmf_net = dnmf.DNMF(rank=2, max_iters=2000, alpha=0.5, output=True, seed=0)
    dnmf_net.fit(Data_matrix=X_train, labels=Y_train, classes=4, label_ratio=1, U_matrix=False)
    V_test = dnmf_net.transform(Data_matrix=X_test)
    scores = dnmf_net.KNN_results(X_train=dnmf_net.V_train, Y_train=Y_train, X_test=V_test, Y_test=Y_test, neighbours=10)
    print(scores)
    # dnmf_net.Draw_KFold() just for 2D