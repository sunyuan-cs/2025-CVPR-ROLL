import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils import TT_split, normalize
import torch
import random
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from scipy.io import loadmat
from decimal import Decimal

def load_data(dataset, neg_prop, aligned_prop, complete_prop, is_noise, NetSeed, test_rate=0):
    all_data = []
    train_pairs = []
    label = []
    

    mat = sio.loadmat('/home/sunyuan/MVC/datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    elif dataset == 'Scene15_NC_0.2':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    elif dataset == 'Scene15_NC_0.5':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    elif dataset == 'Scene15_NC_0.8':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])


    elif dataset == 'nuswide_deep_2_view': #
        data = []
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])
    elif dataset == 'nuswide_deep_2_view_NC_0.2': #_NC_0.2
        data = []
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])
    elif dataset == 'nuswide_deep_2_view_NC_0.5': #_NC_0.2
        data = []
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])
    elif dataset == 'nuswide_deep_2_view_NC_0.8': #
        data = []
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])



    elif dataset == 'xmedia_deep_2_view':
        data = []
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])
    elif dataset == 'xmedia_deep_2_view_NC_0.2':
        data = []
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])
    elif dataset == 'xmedia_deep_2_view_NC_0.5':
        data = []
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])
    elif dataset == 'xmedia_deep_2_view_NC_0.8':
        data = []
        data.append(mat['Img'])
        data.append(mat['Txt'])
        label = np.squeeze(mat['label'])

    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset == 'Reuters_dim10_NC_0.2':
        data = []  # 18758 samples
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    elif dataset == 'Reuters_dim10_NC_0.5':
        data = []  # 18758 samples
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    elif dataset == 'Reuters_dim10_NC_0.8':
        data = []  # 18758 samples
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])

 
    elif dataset == 'CUB-V2':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])  
    elif dataset == 'CUB-V2_NC_0.2':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])  
    elif dataset == 'CUB-V2_NC_0.5':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])  
    elif dataset == 'CUB-V2_NC_0.8':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])  
    # elif dataset == 'LandUse_21':
    #     data = []
    #     label = np.squeeze(mat['Y'])
    #     data.append(sparse.csr_matrix(mat['X'][0, 1]).A)
    #     data.append(sparse.csr_matrix(mat['X'][0, 2]).A) 


    divide_seed = NetSeed
   
   
    formatted_b = format((1 - aligned_prop), '.1f')  # 格式化为字符串
    new_b = float(formatted_b)      # 将字符串转换回浮点数
    train_idx, test_idx = TT_split(len(label), new_b, divide_seed)
    train_label, test_label = label[train_idx], label[test_idx] #取标签，分对齐的+没对齐的
    train_X, train_Y, test_X, test_Y = data[0][train_idx], data[1][train_idx], data[0][test_idx], data[1][test_idx]
    
    if aligned_prop == 1:
        all_data.append(train_X.T)
        all_data.append(train_Y.T)
        all_label, all_label_X, all_label_Y = train_label, train_label, train_label
    else:
        shuffle_idx = random.sample(range(len(test_Y)), len(test_Y))
        test_Y = test_Y[shuffle_idx]
        test_label_X, test_label_Y = test_label, test_label[shuffle_idx] #对模态Y进行打乱，构建没对齐的pair
        all_data.append(np.concatenate((train_X, test_X)).T) #构建得到新的模态X
        all_data.append(np.concatenate((train_Y, test_Y)).T) #构建得到新的模态Y
        all_label = np.concatenate((train_label, test_label)) #原始的标签
        all_label_X = np.concatenate((train_label, test_label_X)) #新的模态X的标签
        all_label_Y = np.concatenate((train_label, test_label_Y)) #新的模态Y的标签

    test_mask = get_sn(2, len(test_label), 1 - complete_prop)
    if aligned_prop == 1.:
        mask = test_mask
    else:
        identy_mask = np.ones((len(train_label), 2))
        mask = np.concatenate((identy_mask, test_mask))

    # pair construction. view 0 and 1 refer to pairs constructed for training. noisy and real labels refer to 0/1 label of those pairs
    if aligned_prop == 1.:
        valid_idx = np.logical_and(mask[:, 0], mask[:, 1])
    else:
        valid_idx = np.ones_like(train_label).astype(np.bool_)
    
    # if test_rate == 0:
    #     test_data = (test_X, test_Y, test_label_X, test_label_Y)
    # el
    if test_rate > 0:
        inx = np.arange(len(test_X))
        np.random.shuffle(inx)
        inx = inx[0: int(len(test_X) * test_rate)]
        test_data = (test_X[inx], test_Y[inx], test_label_X[inx], test_label_Y[inx])
    view0, view1, noisy_labels, real_labels, _, _ = \
        get_pairs(train_X[valid_idx], train_Y[valid_idx], neg_prop, train_label[valid_idx], test_data if aligned_prop < 1 and test_rate > 0 else None)

    # count = 0
    # for i in range(len(noisy_labels)):
    #     if noisy_labels[i] != real_labels[i]:
    #         count += 1
    # print('noise rate of the constructed neg. pairs is ', round(count / (len(noisy_labels) - len(train_X)), 2))

    if is_noise:  # training with noisy negative correspondence
        print("----------------------Training with noisy_labels----------------------")
        train_pair_labels = noisy_labels
        label_aligned = np.concatenate((np.ones_like(train_label), np.zeros_like(test_label))) if aligned_prop < 1 else np.ones_like(train_label)
    else:  # training with gt negative correspondence
        print("----------------------Training with real_labels----------------------")
        train_pair_labels = real_labels
        label_aligned = np.ones_like(all_label)


    train_pairs.append(view0.T)
    train_pairs.append(view1.T)
    train_pair_real_labels = real_labels
    


# # ## 保存噪声关联数据
#     output_dict = {  
#         # 'X1': all_data[0].T,  
#         # 'X2': all_data[1].T,  
#         # 'Y': all_label  

#         'Img': all_data[0].T,  
#         'Txt': all_data[1].T,  
#         'label': all_label  
#     }  
#     print(all_data[0].T.shape)
#     print(all_data[1].T.shape)
#     print(all_label.shape)


#     # 保存输出数据到.mat文件  
#     output_filename = dataset + '_NC_1.0' + '.mat'  
#     sio.savemat(output_filename, output_dict)  
   
#     print('OVER')
#     exit()

    return train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, label_aligned, divide_seed, mask


def get_pairs(train_X, train_Y, neg_prop, train_label, u_data=None):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []
    # construct pos. pairs
    for i in range(len(train_X)):
        view0.append(train_X[i])
        view1.append(train_Y[i])
        labels.append(1)
        real_labels.append(1)
        class_labels0.append(train_label[i])
        class_labels1.append(train_label[i])
    # construct neg. pairs by taking each sample in view0 as an anchor and randomly sample neg_prop samples from view1,
    # which may lead to the so called noisy labels, namely, some of the constructed neg. pairs may in the same category.
    
    train_label_X, train_label_Y = train_label, train_label
    if u_data is not None:
        u_X, u_Y, u_X_labels, u_Y_labels = u_data
        train_X, train_Y, train_label_X, train_label_Y = np.concatenate([train_X, u_X]), np.concatenate([train_Y, u_Y]), np.concatenate([train_label_X, u_X_labels]), np.concatenate([train_label_Y, u_Y_labels])
    for j in range(len(train_X)):
        neg_idx = random.sample(range(len(train_Y)), neg_prop)
        for k in range(neg_prop):
            view0.append(train_X[j])
            view1.append(train_Y[neg_idx[k]])
            labels.append(0)
            class_labels0.append(train_label_X[j])
            class_labels1.append(train_label_Y[neg_idx[k]])
            if train_label_X[j] != train_label_Y[neg_idx[k]]:
                real_labels.append(0)
            else:
                real_labels.append(1)

    labels = np.array(labels, dtype=np.int64)
    real_labels = np.array(real_labels, dtype=np.int64)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int64), np.array(class_labels1, dtype=np.int64)
    view0, view1 = np.array(view0, dtype=np.float32), np.array(view1, dtype=np.float32)
    return view0, view1, labels, real_labels, class_labels0, class_labels1


def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 4.3 of the paper
    :return:Sn
    """
    missing_rate = missing_rate / 2
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix


class getDataset(Dataset):
    def __init__(self, data, labels, real_labels0, real_labels1):
        self.data = data
        self.labels = labels
        self.real_labels0 = real_labels0
        self.real_labels1 = real_labels1

    def __getitem__(self, index):
        fea0, fea1 = (torch.from_numpy(self.data[0][:, index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][:, index])).type(torch.FloatTensor)
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        if len(self.real_labels0) == 0:
            return fea0, fea1, label
        real_label0 = np.int64(self.real_labels0[index])
        real_label1 = np.int64(self.real_labels1[index])
        return fea0, fea1, label, real_label0, real_label1

    def __len__(self):
        return len(self.labels)


class getAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1, mask):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1
        self.mask = mask

    def __getitem__(self, index):
        fea0, fea1 = (torch.from_numpy(self.data[0][:, index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][:, index])).type(torch.FloatTensor)
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        mask = np.int64(self.mask[index])
        return fea0, fea1, label, class_labels0, class_labels1, mask, index

    def __len__(self):
        return len(self.labels)


def loader(train_bs, neg_prop, aligned_prop, complete_prop, is_noise, dataset, test_rate=0):
    """
    :param train_bs: batch size for training, default is 1024
    :param neg_prop: negative / positive pairs' ratio
    :param aligned_prop: known aligned proportions for training SURE
    :param complete_prop: known complete proportions for training SURE
    :param is_noise: training with noisy labels or not, 0 --- not, 1 --- yes
    :param dataset: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing MvCLN
    """
    # train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, \
    # divide_seed, mask = load_data(dataset, neg_prop, aligned_prop, complete_prop, is_noise)
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, label_aligned, divide_seed, mask = load_data(dataset, neg_prop, aligned_prop, complete_prop, is_noise, test_rate=test_rate)
    train_pair_dataset = getDataset(train_pairs, train_pair_labels, train_pair_real_labels, train_pair_real_labels)
    all_dataset = getAllDataset(all_data, all_label, all_label_X, all_label_Y, mask)

    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    all_loader = DataLoader(
        all_dataset,
        batch_size=128,
        shuffle=True
    )
    return train_pair_loader, all_loader, divide_seed


def loader_cl(train_bs, neg_prop, aligned_prop, complete_prop, is_noise, dataset, NetSeed):
    """
    :param train_bs: batch size for training, default is 1024
    :param neg_prop: negative / positive pairs' ratio
    :param aligned_prop: known aligned proportions for training SURE
    :param complete_prop: known complete proportions for training SURE
    :param is_noise: training with noisy labels or not, 0 --- not, 1 --- yes
    :param dataset: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing MvCLN
    """
    
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, label_aligned, divide_seed, mask = load_data(dataset, neg_prop, aligned_prop, complete_prop, is_noise, NetSeed)
    # train_pair_dataset = getDataset(train_pairs, train_pair_labels, train_pair_real_labels)
    # train_pair_dataset = getAllDataset(all_data, label_aligned, all_label_X, all_label_Y, mask)
    

    train_pair_dataset = getAllDataset(all_data, label_aligned, all_label_X, all_label_Y, mask)
    all_dataset = getAllDataset(all_data, all_label, all_label_X, all_label_Y, mask)
    
 
    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    all_loader = DataLoader(
        all_dataset,
        batch_size=128,
        shuffle=True
    )
    return train_pair_loader, all_loader, divide_seed

def get_train_loader(x0, x1, c0, c1, train_bs):
    # train_dataset = getAllDataset([x0.T, x1.T], np.ones_like(c0), c0, c1, randint(1, 2, size=(x0.shape[0], 2)))
    train_dataset = getAllDataset([x0.T, x1.T], np.ones_like(c0), c0, c1, np.ones_like(c0))
    train_pair_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    return train_pair_loader
