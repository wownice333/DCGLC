from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from Utils.evaluate_embedding import cluster_acc


def evaluate_cluster_result(y, y_pred, y_pred_s):
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    acc_s = cluster_acc(y, y_pred)
    nmi_s = nmi_score(y, y_pred)
    ari_s = ari_score(y, y_pred)
    if acc_s>acc:
        acc = acc_s
        nmi=nmi_s
        ari=ari_s
    return acc, nmi, ari