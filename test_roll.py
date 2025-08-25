import argparse
import time
import torch.nn.functional as F
import logging
import os
import sys
import numpy as np
from numpy import *
# import matplotlib.pyplot as plt
from models import *
# from models_ori import *
from Clustering import Clustering
from Clustering_generate import Clustering_generate
from sure_inference import both_infer
from data_loader import loader_cl, get_train_loader
import os
# from kmeans_pytorch import kmeans, kmeans_predict
import math
from sklearn.cluster import KMeans

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score

# import models_ori
# import random
import warnings
os.environ['OMP_NUM_THREADS']="1"


parser = argparse.ArgumentParser(description='MvCLN in PyTorch')
parser.add_argument('--data', default='0', type=int,
                    help='choice of dataset, 0-Scene15, 1-Caltech101, 2-Reuters10, 3-NoisyMNIST,'
                         '4-DeepCaltech, 5-DeepAnimal, 6-MNISTUSPS, 7-WIKI, 8-WIKI-deep, 9-NUSWIDE-deep, 10-xmedia-deep, 11-xrmb')
parser.add_argument('-li', '--log-interval', default='50', type=int, help='interval for logging info')
parser.add_argument('-bs', '--batch-size', default='128', type=int, help='number of batch size')
parser.add_argument('-e', '--epochs', default='100', type=int, help='number of epochs to run')   #100
parser.add_argument('-we', '--warm_epochs', default='50', type=int, help='warmup epochs')
parser.add_argument('-lr', '--learn-rate', default='1e-4', type=float, help='learning rate of adam')
parser.add_argument('--lam', default='0.5', type=float, help='hyper-parameter between losses')
parser.add_argument('-noise', '--noisy-training', type=bool, default=True,
                    help='training with real labels or noisy labels')
parser.add_argument('-np', '--neg-prop', default='30', type=int, help='the ratio of negative to positive pairs')
parser.add_argument('-m', '--margin', default=0.1, type=float, help='initial margin')

parser.add_argument('--q', default=0.1, type=float, help='q parameter')
parser.add_argument('--lamda', default='1e-4', type=float, help='lamda parameter')
parser.add_argument('--alpha', default='1e-4', type=float, help='alpha parameter')
parser.add_argument('--beta', default='1e-2', type=float, help='beta parameter')  

parser.add_argument('--shift', default=7., type=float, help='initial margin')
parser.add_argument('--tau', default=0.5, type=float, help='initial margin')
parser.add_argument('--ratio', default=0, type=float, help='initial test ratio')
parser.add_argument('--gpu', default='1', type=str, help='GPU device idx to use.')
parser.add_argument('-r', '--robust', default=True, type=bool, help='use our robust loss or not')
parser.add_argument('-t', '--switching-time', default=1.0, type=float, help='start fine when neg_dist>=t*margin')
parser.add_argument('-s', '--start-fine', default=False, type=bool, help='flag to start use robust loss or not')
parser.add_argument('--settings', default=2, type=int, help='0-PVP, 1-PSP, 2-Both')
parser.add_argument('-ap', '--aligned-prop', default='1', type=float,
                    help='originally aligned proportions in the partially view-unaligned data')
parser.add_argument('-cp', '--complete-prop', default='1.0', type=float,
                    help='originally complete proportions in the partially sample-missing data')
parser.add_argument('--method', default='pa', type=str, help='initial margin')
parser.add_argument('--test-rate', default=0, type=float, help='initial test rate')
parser.add_argument('--NC', dest='NC', action='store_true', help='noisy correspondencel')
#####################################################################################


warnings.filterwarnings('ignore')
args = parser.parse_args()
print("==========\nArgs:{}\n==========".format(args))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CCL(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, tau=0.5, method='log', q=0.5, ratio=0):
        super(CCL, self).__init__()
        self.tau = tau
        self.method = method
        self.q = q
        self.ratio = ratio

        self.batch_size = args.batch_size
        self.class_num = args.class_num
        self.temperature = tau
        self.device = device

        self.mask = self.mask_correlated_samples(self.batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def forward(self, scores):
        eps = 1e-10
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        randn, eye = torch.rand_like(scores), torch.eye(scores.shape[0]).cuda()
        randn[eye > 0] = randn.min(dim=1)[0] - 1
        n = scores.shape[0]
        num = n - 1 if self.ratio <= 0 or self.ratio >= 1 else int(self.ratio * n)
        V, K = randn.topk(num, dim=1)
        mask = torch.zeros_like(scores)
        mask[torch.arange(n).reshape([-1, 1]).cuda(), K] = 1.

        if self.method == 'log':
            criterion = lambda x: -((1. - x + eps).log() * mask).sum(1).mean()
        elif self.method == 'tan':
            criterion = lambda x: (x.tan() * mask).sum(1).mean()
        elif self.method == 'abs':
            criterion = lambda x: (x * mask).sum(1).mean()
        elif self.method == 'exp':
            criterion = lambda x: ((-(1. - x)).exp() * mask).sum(1).mean()
        elif self.method == 'gce':
            criterion = lambda x: ((1. - (1. - x + eps) ** self.q) / self.q * mask).sum(1).mean()
        elif self.method == 'infoNCE':
            criterion = lambda x: -x.diag().log().mean()
        else:
            raise Exception('Unknown Loss Function!')
        return criterion(i2t) + criterion(t2i)
    
    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) +(p_i * torch.log(p_i)).sum() 
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) +(p_j * torch.log(p_j)).sum()  
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.tau
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + entropy

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

def best_map(y_true, y_pred):
    """
    使用匈牙利算法将聚类标签重新排列以匹配真实标签。
    
    参数:
    y_true -- 真实标签 (1D array)
    y_pred -- 聚类标签 (1D array)
    
    返回:
    new_y_pred -- 重新排列后的聚类标签 (1D array)
    """
    # 找到标签的唯一值
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # 创建一个混淆矩阵
    confusion_matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=np.int32)
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            confusion_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    # 创建一个字典进行映射
    mapping = {unique_pred[col]: unique_true[row] for row, col in zip(row_ind, col_ind)}
    
    # 重新排列聚类标签
    new_y_pred = np.array([mapping[label] for label in y_pred])
    
    return new_y_pred
def kmeans(x, initial_centers, num_clusters, max_iters=100):
    centers = initial_centers
    for i in range(max_iters):
        # Calculate distances between each point and the centers
        distances = torch.cdist(x, centers)     
        # Assign each point to the nearest center
        labels = torch.argmin(distances, dim=1)      
        # Calculate new centers
        new_centers = torch.stack([x[labels == j].mean(dim=0) for j in range(num_clusters)])     
        # Check for convergence (if centers do not change)
        if torch.allclose(centers, new_centers, atol=1e-4):
            break       
        centers = new_centers  
    return centers



def contrastive_train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    time0 = time.time()
    for batch_idx, (x0, x1, labels, real_labels_X, real_labels_Y, mask, index) in enumerate(train_loader):
        x0, x1, labels, real_labels = x0.to(device), x1.to(device), labels.to(device), real_labels_Y.to(device)
        if not args.NC:
            x0, x1 = x0[labels > 0], x1[labels > 0]      
        x0 = x0.view(x0.size()[0], -1)
        x1 = x1.view(x1.size()[0], -1)
        try:
            h0, h1, z0, z1 = model(x0, x1)
        except:
            print("error raise in batch", batch_idx)
        loss1 = criterion[0](x0, z0)+criterion[0](x1, z1)
        loss_cl=lambda x1,x2,M: -(M * (x1.mm(x2.t())/args.tau).softmax(1).log()).sum(1).mean()
        I = torch.eye(x0.size(0)).cuda()
        loss2 = (args.lamda * loss_cl(h0, h1, I))
        loss = loss1 + loss2 
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_time = time.time() - time0
    return epoch_time



loss_CE = nn.CrossEntropyLoss()

def train(train_loader, model, criterion, optimizer, epoch, args, class_centers):

    if epoch % args.log_interval == 0:
        logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))
    model.train()
    time0 = time.time()
    ncl_loss_value = 0
    ver_loss_value = 0
    cur_final_centers_list = []
    for batch_idx, (x0, x1, labels, real_labels_X, real_labels_Y, mask,index) in enumerate(train_loader):
   
        x0, x1, labels, real_labels = x0.to(device), x1.to(device), labels.to(device), real_labels_Y.to(device)
        p0 = train_loader.pseudo_label0[index].to(device)
        p1 = train_loader.pseudo_label1[index].to(device)

        if not args.NC:
            x0, x1 = x0[labels > 0], x1[labels > 0]
            p0 = train_loader.pseudo_label0[index[labels>0]].to(device)
            p1 = train_loader.pseudo_label1[index[labels>0]].to(device)
        p0 = p0.to(dtype=torch.float)
        p1 = p1.to(dtype=torch.float)
        x0 = x0.view(x0.size()[0], -1)
        x1 = x1.view(x1.size()[0], -1) 
        try:
            h0, h1, z0, z1 = model(x0, x1)
        except:
            print("error raise in batch", batch_idx)
        loss1 = (criterion[0](x0, z0)+criterion[0](x1, z1))
        final_centers = torch.tensor(class_centers).to(device) 
        zp0 = torch.mm(h0, (final_centers.T))
        zp1 = torch.mm(h1, (final_centers.T))     
        pre0 = F.softmax(zp0, dim=0) 
        pre1 = F.softmax(zp1, dim=0) 
        sim = pre0.mm(pre1.t())
        diag = sim.diag()
        w = diag / diag.sum().detach()
        pp0 = p0.mm(pre0.t().log()).sum(1)
        lossa = - (args.beta * w * pp0).mean()                
        pp1 = p1.mm(pre1.t().log()).sum(1)
        lossb = - (args.beta * w * pp1).mean()
        loss2 = lossa+lossb
        cos = h0.mm(h1.t())
        sim = (cos / args.tau).exp()
        pos = sim.diag()
        q = args.q
        loss_rcl = (((1 - q) * (sim.sum(1))**q -  pos ** q) / q).sum().mean()
        loss3 = args.alpha * loss_rcl
        loss = loss1 + loss2 + loss3
        ncl_loss_value += 0
        ver_loss_value += 0
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_time = time.time() - time0


    if epoch % args.log_interval == 0:
        logging.info("dist: time = {} s".format(round(epoch_time, 2)))

    return epoch_time, cur_final_centers_list





def main(): 
    #choose the dataset      
                                                                 
    data_name = ['Scene15_NC_0.2', 'Reuters_dim10_NC_0.2', 'nuswide_deep_2_view_NC_0.2', 'xmedia_deep_2_view_NC_0.2', 'CUB-V2_NC_0.2']
    # data_name = ['Scene15_NC_0.5', 'Reuters_dim10_NC_0.5', 'nuswide_deep_2_view_NC_0.5', 'xmedia_deep_2_view_NC_0.5', 'CUB-V2_NC_0.5']
    # data_name = ['Scene15_NC_0.8', 'Reuters_dim10_NC_0.8', 'nuswide_deep_2_view_NC_0.8', 'xmedia_deep_2_view_NC_0.8', 'CUB-V2_NC_0.8']
    # data_name = ['Scene15', 'Reuters_dim10', 'nuswide_deep_2_view', 'xmedia_deep_2_view', 'CUB-V2']



    NetSeed_list = [1111,2222,3333,4444,5555]           
    NetSeed = NetSeed_list[i]
    np.random.seed(NetSeed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(NetSeed)
    torch.cuda.manual_seed(NetSeed) 
    train_pair_loader, all_loader, _ = loader_cl(args.batch_size, args.neg_prop, args.aligned_prop,
                                                        args.complete_prop, args.noisy_training,
                                                        data_name[args.data], NetSeed)
    
 



    # args.lamda = 1e-4 
    # args.alpha = 1e-4 
    # args.beta = 1e-2 
    # args.q = 0.1

    if args.data == 0:
        model = SUREfcScene().to(device)
        args.class_num = 15 
        args.learn_rate = 1e-3
    elif args.data == 1:
        model = SUREfcReuters().to(device)
        args.class_num = 6
        args.learn_rate = 1e-4
    elif args.data == 2:
        model = SUREfcnuswidedeep().to(device)
        args.class_num = 10 
        args.learn_rate = 1e-4
    elif args.data == 3:
        model = SUREfcxmediadeep().to(device)
        args.class_num = 200 
        args.learn_rate = 1e-5
    elif args.data == 4:
        model = SUREfccub().to(device)
        args.class_num = 10
        args.learn_rate = 1e-4




    valid_datasets = ['Scene15', 'Reuters_dim10', 'nuswide_deep_2_view', 'xmedia_deep_2_view', 'CUB-V2']    
    args.epochs = 200 if data_name[args.data] in valid_datasets else args.epochs 
    criterion_mse = nn.MSELoss().to(device)
    criterion_ccl = CCL(tau=args.tau, method=args.method, q=args.margin, ratio=args.ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.info("******** Training begin ********")
    

    print('Warmup begining')
    print('...') 


    for epoch in range(0, args.warm_epochs):

        contrastive_train(train_pair_loader, model, [criterion_mse, criterion_ccl], optimizer, epoch, args)    
        if epoch == args.warm_epochs-1: 
            v0, v1, gt_label = both_infer(model, device, all_loader, args.settings, cri=None, return_data=False)
            data = [v0, v1]
            class_centers, pseudo_label0, pseudo_label1 = Clustering_generate(data, gt_label)   
    print('Warmup finished')  

    index_total_pred0 = torch.tensor(pseudo_label0).long()
    index_total_pred1 = torch.tensor(pseudo_label1).long() 
    train_pair_loader.pseudo_label0=index_total_pred0
    train_pair_loader.pseudo_label1=index_total_pred1
    for epoch in range(0, args.epochs):
        
        train(train_pair_loader, model, [criterion_mse, criterion_ccl], optimizer, epoch, args, class_centers)
        if epoch == args.epochs-1:   
            print('epoch: ', epoch)

  
            v0, v1, gt_label = both_infer(model, device, all_loader, args.settings, cri=True, return_data=False)      
            data = [v0, v1]
            ret = Clustering(data, gt_label)  
            
            
            logging.info("Clustering: alpha={}, beta={}, q= {}, aligned_prop= {}, learn_rate= {}, acc={}, nmi={}, ari={}".format(args.alpha, args.beta, args.q, args.aligned_prop, args.learn_rate, ret['kmeans']['accuracy'], ret['kmeans']['NMI'], ret['kmeans']['ARI']))  


    return ret['kmeans']['accuracy'], ret['kmeans']['NMI'], ret['kmeans']['ARI']







result_acc = []
result_nmi = []
result_ari = []
if __name__ == '__main__':
    for i in range(5):
        acc, nmi, ari = main()
        result_acc.append(acc)
        result_nmi.append(nmi)
        result_ari.append(ari)

    acc_mean = np.mean(result_acc)
    nmi_mean = np.mean(result_nmi)
    ari_mean = np.mean(result_ari)


    logging.info("Clustering 5 times: lambda= {}, alpha={}, q= {}, aligned_prop= {}, learn_rate= {}, acc={}, nmi={}, ari={}".format(args.beta, args.alpha, args.q, args.aligned_prop, args.learn_rate, acc_mean, nmi_mean, ari_mean))  

    with open('result.txt', 'a+') as f:
        f.write('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(
        args.beta, args.alpha, args.q, args.learn_rate, acc_mean, nmi_mean, ari_mean))
        f.flush() 