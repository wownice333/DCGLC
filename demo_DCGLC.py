from tqdm import tqdm

from Utils.InitializeD import Initialization_D
import warnings
import scipy

warnings.filterwarnings("ignore")

from arguments import arg_parse
from torch_geometric.data import DataLoader
from Utils.aug import TUDataset_aug as TUDataset
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from Utils.evaluate_embedding import cluster_acc
from model.model import DCGLC
import scipy


def refined_subspace_affinity(s):
    s=torch.tensor(s)
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    args = arg_parse()
    accuracies = {'acc': [], 'nmi': [], 'ari': [], 'randomforest': []}
    epochs = 100
    log_interval = 1
    batch_size = 16
    lr = args.lr
    # DS = args.DS
    print(args.aug)
    for DS in ['BZR','AIDS','DD']:#
        args.DS=DS
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', DS)
        dataset = TUDataset(path, name=DS, aug=args.aug)#.shuffle()
        dataset_eval = TUDataset(path, name=DS, aug='none')#.shuffle()
        n_cluster = len(np.unique(dataset.data.y))
        dataset_num_features = max(dataset.data.num_features, 1)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

        args.cluster_emb = args.d * n_cluster

        print('================')
        print('Dataset:', DS)
        print('lr: {}'.format(lr))
        print('num_features: {}'.format(dataset_num_features))
        print('hidden_dim: {}'.format(args.hidden_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))
        print('clutering embedding dimension: {}'.format(args.cluster_emb))
        print('================')
        model = DCGLC(args, dataset_num_features, n_cluster, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mode = 'fd'
        measure = 'JSD'
        if args.eval:
            model.load_state_dict(torch.load("./weight/" + args.DS + "_model.pth"))
            model.eval()
            emb, tmp_q, tmp_s, y = model.get_results(dataloader)
            # y_pred_s = tmp_s.argmax(1)
            # y_pred = tmp_q.argmax(1)
            tmp_total = np.maximum(tmp_s, tmp_q)
            y_pred = tmp_total.argmax(1)

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)

            accmax = acc
            nmimax = nmi
            arimax = ari
            print(accmax)
            with open('./result/' + args.DS + '_result.txt', 'a') as f:
                f.write('Load Saved .pth File' + '\n')
                f.write(args.DS + '_Result:' + '\n')
                f.write('ACC: {:.2f} (0.00)\n'.format(accmax*100))
                f.write('NMI: {:.2f} (0.00)\n'.format(nmimax*100))
                f.write('ARI: {:.2f} (0.00)\n'.format(arimax*100))
                f.write('\n')

            print('Load Saved .pth File for '+ args.DS + '\n')
            print('ACC: {:.2f} (0.00)\n'.format(accmax*100))
            print('NMI: {:.2f} (0.00)\n'.format(nmimax*100))
            print('ARI: {:.2f} (0.00)\n'.format(arimax*100))
        else:
            para_set = [0.001,0.01,0.1,1,10,100]
            for lamda in para_set:
                args.lamda = lamda
                for beta in para_set:
                    args.beta = beta
                    iter = 1
                    ACCList = np.zeros((iter, 1))
                    NMIList = np.zeros((iter, 1))
                    ARIList = np.zeros((iter, 1))
                    ACC_MEAN = np.zeros((1, 2))
                    NMI_MEAN = np.zeros((1, 2))
                    ARI_MEAN = np.zeros((1, 2))

                    for it in range(iter):
                        acc = -1.
                        nmi = -1.
                        ari = -1.
                        accmax = -1.
                        nmimax = -1.
                        arimax = -1.
                        aug_P = np.ones(5) / 5
                        history_loss = []
                        pbar = tqdm(range(1, epochs + 1))
                        for epoch in pbar:
                            dataloader.dataset.aug_P = aug_P
                            loss_all = 0
                            batch = 0
                            n_aug = np.random.choice(5, 1, p=aug_P)[0]

                            if epoch == 1:
                                model.eval()
                                emb, _, _, y = model.get_results(dataloader)
                                kmeans = KMeans(n_clusters=n_cluster, n_init=100)
                                y_pred = kmeans.fit_predict(emb)
                                model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
                                # Initialize D
                                D = Initialization_D(emb, y_pred, n_cluster, args.d)
                                D = torch.tensor(D).to(torch.float32)
                                model.D.data = D.to(device)

                            if epoch % log_interval == 0:
                                model.eval()
                                emb, tmp_q, tmp_s, y = model.get_results(dataloader)

                                tmp_total = np.maximum(tmp_s, tmp_q)
                                y_pred = tmp_total.argmax(1)

                                acc = cluster_acc(y, y_pred)
                                nmi = nmi_score(y, y_pred)
                                ari = ari_score(y, y_pred)

                                s_tilde = refined_subspace_affinity(tmp_s).to(device)
                                p = target_distribution(torch.tensor(tmp_q)).to(device)
                                if acc > accmax:
                                    accmax = acc
                                    torch.save(model.state_dict(), './weight/' + args.DS +'_'+str(acc)+'_model.pth')
                                if nmi > nmimax:
                                    nmimax = nmi
                                if ari > arimax:
                                    arimax = ari
                            print(accmax)

                            model.train()
                            for idx, data in enumerate(dataloader):
                                data, data_aug = data
                                batch_idx=np.unique(data.batch)+batch_size*idx
                                data=data.to(device)
                                data_aug=data_aug.to(device)

                                optimizer.zero_grad()
                                node_num, _ = data.x.size()
                                z, q, s, x = model(data.x, data.edge_index, data.batch, data.num_graphs)

                                if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4' or args.aug == 'minmax':
                                    edge_idx = data_aug.edge_index.cpu().numpy()
                                    _, edge_num = edge_idx.shape
                                    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                                    node_num_aug = len(idx_not_missing)
                                    data_aug.x = data_aug.x[idx_not_missing]

                                    data_aug.batch = data.batch[idx_not_missing]
                                    idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                                    edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                                                not edge_idx[0, n] == edge_idx[1, n]]
                                    data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

                                z_aug, q_aug, s_aug, x_aug = model(data_aug.x, data_aug.edge_index.to(device), data_aug.batch, data_aug.num_graphs)
                                loss = model.loss_cal(z, z_aug)
                                loss = loss + args.lamda * model.loss_cal(q, s) #+args.nu * loss_balance_entropy(q)

                                if epoch >= 1:
                                    # Subspace clustering loss
                                    kl_loss = F.kl_div(q.log(), p[batch_idx])
                                    kl_loss_sub = F.kl_div(s.log(), s_tilde[batch_idx])
                                    loss = loss + args.beta*(kl_loss + kl_loss_sub)
                                    batch += 1
                                else:
                                    loss = loss
                                # Total loss
                                loss_all += loss.item()
                                loss.backward()

                                optimizer.step()
                            history_loss.append(loss_all)

                            # minmax
                            loss_aug = np.zeros(5)
                            for n in range(5):
                                _aug_P = np.zeros(5)
                                _aug_P[n] = 1
                                dataloader.dataset.aug_P = _aug_P
                                count, count_stop = 0, len(dataloader) // 5 + 1
                                with torch.no_grad():
                                    for data in dataloader:

                                        data, data_aug = data
                                        node_num, _ = data.x.size()
                                        data = data.to(device)
                                        z, _, _, x = model(data.x, data.edge_index, data.batch, data.num_graphs)

                                        if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4' or args.aug == 'minmax':
                                            edge_idx = data_aug.edge_index.numpy()
                                            _, edge_num = edge_idx.shape
                                            idx_not_missing = [n for n in range(node_num) if
                                                            (n in edge_idx[0] or n in edge_idx[1])]

                                            node_num_aug = len(idx_not_missing)
                                            data_aug.x = data_aug.x[idx_not_missing]

                                            data_aug.batch = data.batch[idx_not_missing]
                                            idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                                            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in
                                                        range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                                            data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

                                        data_aug = data_aug.to(device)
                                        z_aug, _, _, x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch,
                                                            data_aug.num_graphs)

                                        loss = model.loss_cal(z, z_aug)
                                        loss_aug[n] += loss.item() * data.num_graphs
                                        if args.mode == 'fast':
                                            count += 1
                                            if count == count_stop:
                                                break

                                if args.mode == 'fast':
                                    loss_aug[n] /= (count_stop * batch_size)
                                else:
                                    loss_aug[n] /= len(dataloader.dataset)

                            gamma = float(args.gamma)
                            epsilon = 1
                            b = aug_P + epsilon * (loss_aug - gamma * (aug_P - 1 / 5))

                            mu_min, mu_max = b.min() - 1 / 5, b.max() - 1 / 5
                            mu = (mu_min + mu_max) / 2
                            # bisection method
                            while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
                                if np.maximum(b - mu, 0).sum() > 1:
                                    mu_min = mu
                                else:
                                    mu_max = mu
                                mu = (mu_min + mu_max) / 2

                            aug_P = np.maximum(b - mu, 0)
                            aug_P /= aug_P.sum()

                            pbar.set_description(
                                "Epoch{}| # Total Loss: {:.4}".format(
                                    epoch,
                                    loss_all / len(dataloader),
                                )
                            )
                        ACCList[it - 1, :] = accmax
                        NMIList[it - 1, :] = nmimax
                        ARIList[it - 1, :] = arimax

            

                    ACC_MEAN[0, :] = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
                    NMI_MEAN[0, :] = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
                    ARI_MEAN[0, :] = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)


                    with open('./result/' + args.DS + '_result.txt', 'a') as f:
                        f.write(args.DS + '_Result:' + '\n')
                        f.write('Contrastive loss:'+ str(args.lamda)  + '\n')
                        f.write('Clustering loss:'+ str(args.beta)  + '\n')
                        f.write('ACC_MEAN:' + str(ACC_MEAN[0][0]*100)+' ('+str(ACC_MEAN[0][1]*100) + ')\n')
                        f.write('NMI_MEAN:' + str(NMI_MEAN[0][0]*100)+' ('+str(NMI_MEAN[0][1]*100) + ')\n')
                        f.write('ARI_MEAN:' + str(ARI_MEAN[0][0]*100)+' ('+str(ARI_MEAN[0][1]*100) + ')\n')
                        f.write('\n')

                    print('ACC:\n' + str(ACC_MEAN))
                    print('NMI:\n' + str(NMI_MEAN))
                    print('ARI:\n' + str(ARI_MEAN))

   