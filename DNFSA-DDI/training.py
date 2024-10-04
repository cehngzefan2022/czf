import random
from datetime import datetime
import time
import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.nn import Node2Vec

from models.ours import GATNet
# from models.FocalLoss import focalLoss
from models.deepddi import DeepDNN
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, \
    balanced_accuracy_score
from sklearn import metrics
import pandas as pd


# import matplotlib.pyplot as plt

# training function at each epoch
# def train(model, device, drug1_loader_train, drug2_loader_train, cid_feature, optimizer, epoch):
def train(model, device, drug1_loader_train, drug2_loader_train, cid_feature, link_edge_index_only1, optimizer,
          epoch):  # deepddi
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        # start = time.time()
        data1 = data[0]
        data2 = data[1]

        # 全部
        subnet_edges = link_edge_index_only1
        # #按批次构图
        # subnet_edges = []
        # for i, (y1, y2) in enumerate(zip(data1.y, data2.y)):
        #     if y1 == 1 and y2 == 1:
        #         subnet_edges.append((data1.z[i], data2.z[i]))
        # subnet_edges = torch.tensor(subnet_edges).T

        # data3 = data[2]
        # data3 = torch.stack(data3, dim=0)

        data1 = data1.to(device)
        data2 = data2.to(device)
        subnet_edges = subnet_edges.to(torch.long).to(device)
        # Net_edges_train = data3.to(device)

        cid_feature = cid_feature.to(device)

        y = data[0].y.view(-1, 1).to(device)
        y = y.squeeze(1).long()

        # output = model(data1 ,data2)
        output = model(data1, data2, cid_feature, subnet_edges)

        # output = model(data1, data2)#deepddi
        # output=output.view(-1, 1)#.squeeze(1)
        # output = output.to(torch.float32)
        loss = loss_fn(output, y)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test, cid_feature, link_edge_index_only1):
    # def predicting(model, device, drug1_loader_test, drug2_loader_test):#deepddi
    # model.load_state_dict(torch.load('data/result/DDI_612/6/1--model_DDI_612.model'))
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            # data3 = data[2]
            # data3 = torch.stack(data3, dim=0)

            # 全部
            subnet_edges = link_edge_index_only1
            # #按批次构图
            # subnet_edges = []
            # for i, (y1, y2) in enumerate(zip(data1.y, data2.y)):
            #     if y1 == 1 and y2 == 1:
            #         subnet_edges.append((data1.z[i], data2.z[i]))
            # subnet_edges = torch.tensor(subnet_edges).T

            data1 = data1.to(device)
            data2 = data2.to(device)
            subnet_edges = subnet_edges.to(torch.long).to(device)

            # Net_edges_test = data3.to(device)
            cid_feature = cid_feature.to(device)

            output = model(data1, data2, cid_feature, subnet_edges)
            # output = torch.sigmoid(output)

            # output = model(data1, data2)#deepddi

            ys = F.softmax(output, 1).to('cpu').data.numpy()

            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            # 使用 sigmoid 激活函数
            # ys = output.to('cpu').data.numpy()

            # # 预测类别（0或1）
            # predicted_labels = list(map(lambda x: 1 if x > 0.5 else 0, ys))
            # # 获取类别为1的概率（sigmoid 输出值）
            # predicted_scores = ys

            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


# modeling = torch.load('data/result/DDI_612/6/1--model_DDI_612.model')#预训练模型
modeling = GATNet
# modeling = DeepDNN

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 50
NUM_EPOCHS = 250

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
# datafile = 'DDI_612'  #train
# drug_num =612
# datafile = '612_TWOSIDE1'
# datafile = 'test_S1'
datafile = 'STNN01'  # train
drug_num = 555

# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda:1')  ############################ GPU
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


def creat_link_data(datafile):
    # df = pd.read_csv('data/' + datafile + '.csv')
    # df_drug_list = pd.read_csv('data/DDI_613_SMILE.csv')
    # 构建关联网络
    df = pd.read_csv('data/STNN/' + datafile + '.csv')
    df_drug_list = pd.read_csv('data/STNN/DrugID_DrugCID_DrugName.csv')

    idx = df_drug_list['drugID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}  # 每个药物标识符分配一个整数索引i

    df_data_t = df[df.label == 1]

    edges_only1 = df_data_t[['drug1', 'drug2']].values
    edges = df[['drug1', 'drug2']].values
    edges_idx_only1 = [[idx_map[e[0]], idx_map[e[1]]] for e in edges_only1]
    edges_idx = [[idx_map[e[0]], idx_map[e[1]]] for e in edges]

    # G = nx.Graph(edges_idx).to_directed()  # 使用边列表创建有向图
    # link_edge_index = []
    # for e1, e2 in G.edges:  # 遍历有向图中的每条边
    #     link_edge_index.append([e1, e2])
    # # link_edge_index= [list(item) for item in link_edge_index]
    # link_edge_index_Tensor = torch.tensor(link_edge_index).T

    edges1_idx_tensor = torch.tensor(edges_idx_only1).T
    edges_idx_tensor = torch.tensor(edges_idx).T

    # node2vec
    node2vec = Node2Vec(edges1_idx_tensor, embedding_dim=256, walk_length=20, context_size=10)
    link_feature = node2vec.embedding.weight
    link_feature = F.normalize(link_feature)

    cid_list = (df_drug_list['CID'])
    cid_feature = torch.zeros((drug_num, 881))
    for i in range(len(cid_list)):
        string = cid_list[i]
        tensor_row = torch.tensor([int(char) for char in string])
        cid_feature[i, :] = tensor_row

    '''# smile_graph = {}  # 建空字典
    # with open('data/unique.csv', 'r') as file:
    #     reader = csv.reader(file)
    #     data = list(zip(*reader))  # 转置操作，将行转换为列
    #     Net_feature = []
    #     for i in range(1,len(data[0])):
    #         name = data[0][i]
    #         smile = data[1][i]
    #         print(name, smile)
    #         g = smile_to_graph(smile)  # 将SMILES串转换为图形结构 #g = 这个smile串对应的原子数目，特征数目和边的索引
    #         Net_feature.append(g[3])
    # # 索引边
    # Net_edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    # # adj = sp.coo_matrix((np.ones(Net_edges.shape[0]), (Net_edges[:, 0], Net_edges[:, 1])),
    # #                     shape=(len(idx), len(idx)),
    # #                     dtype=np.float32)'''

    return link_feature, cid_feature, edges1_idx_tensor, edges_idx  # 药物分子特征、分子间邻接矩阵


link_feature, cid_feature, link_edge_index_only1, link_edge_index = creat_link_data(datafile)

drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1_STNN78256')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2_STNN78256')  # 78+178


# drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1_both_noskip')
# drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2_both_noskip')#78+178
# drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1_TWOSIDE1')
# drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2_TWOSIDE1')
# drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1_deepddi_PCA')
# drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2_deepddi_PCA')

# drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1_256_8')#S1_data
# drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2_256_8')

#############ablation
# drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1_node2vec_noRDKit')
# drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2_node2vec_noRDKit')

# drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1_RDKit_no2vec')
# drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2_RDKit_no2vec')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.BCELoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


'''#####################################test
#drug
drug1_loader_test = DataLoader(drug1_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
drug2_loader_test = DataLoader(drug2_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
Net_edges_test = torch.LongTensor(link_edge_index).transpose(1, 0)
Net_edges_test_loder = DataLoader(Net_edges_test, batch_size=115,shuffle=None)

model = modeling().to(device)
file_AUCs = 'data/result/twoside/'+'--AUCs--'+datafile+'.txt'
result_file_name = 'data/result/twoside/independent/'+datafile+'.csv'
best_auc = 0
T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test, link_feature, cid_feature, Net_edges_test)

# compute preformence
# AUC = roc_auc_score(T, S)
# precision, recall, threshold = metrics.precision_recall_curve(T, S)
# PR_AUC = metrics.auc(recall, precision)
# BACC = balanced_accuracy_score(T, Y)
# tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
# TPR = tp / (tp + fn)
# PREC = precision_score(T, Y)
ACC = accuracy_score(T, Y)
# KAPPA = cohen_kappa_score(T, Y)
# recall = recall_score(T, Y)

# independent_num = []
# independent_num.append(lenth)
# independent_num.append(T)
# independent_num.append(Y)
# independent_num.append(S)
# txtDF = pd.DataFrame(data=independent_num)
# txtDF.to_csv(result_file_name, index=False, header=False)

# save data
if best_auc < ACC:
    best_auc = ACC
    # AUCs = [AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall, LR]
    # save_AUCs(AUCs, file_AUCs)
print('best_auc:-------', best_auc)

'''

# balance data
y1 = drug1_data.y
# 获取0和1的索引
zero_indices = torch.nonzero(y1 == 0).view(-1)
one_indices = torch.nonzero(y1 == 1).view(-1)

# 从0和1的索引中随机选择相同数量的索引
random_zero_indices = random.sample(zero_indices.tolist(), len(one_indices))
random_one_indices = random.sample(one_indices.tolist(), len(one_indices))

# 合并选择的索引
random_num1 = random_zero_indices + random_one_indices
# random_num1 = random.sample((random_zero_indices + random_one_indices),len(random_num0))
# random_num2 = random_one_indices

lenth = len(random_num1)
random_num1 = random.sample(range(0, lenth), lenth)  # 打乱数据集中样本的顺序
# random_num2 = random.sample(range(0, len(link_edge_index_only1)),len(link_edge_index_only1))
pot1 = int(len(random_num1) / 5)
# pot2 = int(len(random_num2)/5)
print('lenth', lenth)
print('pot', pot1)
# k_fold = 5
# # 计算每折的大小
# fold_size = lenth // k_fold
# # 定义五个折的索引范围
# fold_indices = [slice(i * fold_size, (i + 1) * fold_size) for i in range(k_fold)]
all_ACC = 0
all_AUC = 0
all_PR_AUC = 0
all_F1 = 0
all_PREC = 0
all_AP = 0
all_RACELL = 0
for i in range(5):

    # NUM_EPOCHS = NUM_EPOCHS+20
    # link
    # Net_edges_test = [link_edge_index[idx] for idx in random_num1[pot2*i:pot2*(i+1)]]
    # Net_edges_train = [link_edge_index[idx] for idx in (random_num1[:pot2*i] + random_num1[pot2*(i+1):])]

    # Net_edges_train = torch.LongTensor(Net_edges_train).transpose(1, 0)
    # Net_edges_test = torch.LongTensor(Net_edges_test).transpose(1, 0)
    # Net_edges_train_loder = DataLoader(Net_edges_train,batch_size=TRAIN_BATCH_SIZE,shuffle=None)
    # Net_edges_test_loder = DataLoader(Net_edges_test, batch_size=TRAIN_BATCH_SIZE,shuffle=None)

    # drug
    test_num = random_num1[pot1 * i:pot1 * (i + 1)]
    train_num = random_num1[:pot1 * i] + random_num1[pot1 * (i + 1):]
    # # 从 train_num 中取前3000个序列作为 val_num
    # val_num = train_num[:len(test_num)]

    # # 从 train_num 中取后面的部分作为新的 train_num
    # train_num = train_num[len(test_num):]

    print(len(train_num), len(test_num))

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    # drug1_data_val = drug1_data[val_num]
    # drug1_loader_val = DataLoader(drug1_data_val, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    # drug2_data_val = drug2_data[val_num]
    # drug2_loader_val = DataLoader(drug2_data_val, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCELoss()
    # loss_fn = FocalLoss(gamma=2, weight=None)#####Focal Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=0.0005)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    # deepddi
    # file_AUCs = 'data/result/deepddi/drug/5/1'+str(i)+'--AUCs--'+datafile+'.txt'
    # file_AUCs = 'data/result/deepddi/twoside/4/'+str(i)+'--AUCs--'+datafile+'.txt'

    ##ours
    # file_AUCs = 'data/result/ours/drug/3.14/'+str(i)+'--AUCs--'+datafile+'.txt'
    file_AUCs = 'data/result/ours/stnn/ablation/5.3/' + 'AUCs--wangluo' + datafile + str(i) + '.txt'

    # model_file_name = 'ddata/result/ours/stnn/3.8/'+'model_1' + str(i)+datafile+'.pth'
    # file_AUCs = 'data/result/ours/twoside/5(nocid)/2/'+str(i)+'--AUCs--'+datafile+'.txt'
    # model_file_name = 'data/result/ours/twoside/1/'+str(i)+'--model_' +datafile+'.model'
    # result_file_name = 'data/result/ours/twoside/txtDF/1/'+str(i)+'--result_'+datafile+'.csv'

    # model_file_name = 'data/result/ours/drug/78/256_8/'+'--model_' +datafile+str(i)+'.model'
    # result_file_name = 'data/result/ours/drug/78/256_8/'+'--result_'+datafile+str(i)+'.csv'

    # output_file_path = 'data/result/ours/drug/78/256_8/resoutput_file.txt'
    # file_AUCs = 'data/result/DDI_612/TWOSIDE/5/'+str(i)+'--AUCs--'+datafile+'.txt'
    # AUC_map = 'data/result/picture/NEW1/GATNet(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile + '.png'
    # AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    AUCs = ('epoch\tACC\tAUC\tPR_AUC\tF1\tPREC\trecall\tAP')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    # model_file_name = 'data/result/deepddi/'+str(i)+'--model_' +datafile+'.model'
    # #result_file_name = 'data/result/txtDF/6/'+str(i)+'--result_'+datafile+'.csv'
    # file_AUCs = 'data/result/deepddi/'+str(i)+'--AUCs--'+datafile+'.txt'
    # #AUC_map = 'data/result/picture/NEW1/GATNet(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile + '.png'
    # AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    # with open(file_AUCs, 'w') as f:
    #     f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        # train(model, device, drug1_loader_train, drug2_loader_train, cid_feature, optimizer, epoch + 1)
        # T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test, cid_feature)
        train(model, device, drug1_loader_train, drug2_loader_train, cid_feature, link_edge_index_only1, optimizer,
              epoch + 1)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test, cid_feature, link_edge_index_only1)
        # T is correct label
        # S is predict score
        # Y is predict label

        # compute preformence
        ACC = metrics.accuracy_score(T, Y)
        AUC = metrics.roc_auc_score(T, S)
        F1 = metrics.f1_score(T, Y)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        # BACC = balanced_accuracy_score(T, Y)
        # tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        # TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        AP = metrics.average_precision_score(T, S)
        # KAPPA = cohen_kappa_score(T, Y)
        RACELL = recall_score(T, Y)

        # Append metrics to lists

        # save data
        if best_auc < AUC:
            best_auc = AUC
            AUCs = [epoch, ACC, AUC, PR_AUC, F1, PREC, RACELL, AP]
            save_AUCs(AUCs, file_AUCs)
            # torch.save(model.state_dict(), model_file_name)
            # independent_num = []
            # independent_num.append(test_num)
            # independent_num.append(T)
            # independent_num.append(Y)
            # independent_num.append(S)
            # txtDF = pd.DataFrame(data=independent_num)
            # txtDF.to_csv(result_file_name, index=False, header=False)
            j = 0
        else:
            j = j + 1
        print('best_auc:-------', best_auc)

        if j >= 20:
            all_ACC = all_ACC + ACC
            all_AUC = all_AUC + AUC
            all_PR_AUC = all_PR_AUC + PR_AUC
            all_F1 = all_F1 + F1
            all_PREC = all_PREC + PREC
            all_AP = all_AP + AP
            all_RACELL = all_RACELL + RACELL
            break

    if i == 4:
        avg_ACC = all_ACC / 5
        avg_AUC = all_AUC / 5
        avg_PR_AUC = all_PR_AUC / 5
        avg_F1 = all_F1 / 5
        avg_PREC = all_PREC / 5
        avg_AP = all_AP / 5
        avg_RACELL = all_RACELL / 5
        print("Average ACC:", avg_ACC)
        print("Average AUC:", avg_AUC)
        print("Average PR AUC:", avg_PR_AUC)
        print("Average F1:", avg_F1)
        print("Average Precision:", avg_PREC)
        print("Average AP:", avg_AP)
        print("Average Recall:", avg_RACELL)
        # # model.load_state_dict(torch.load(model_file_name)) # 独立测试
        # T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test, cid_feature, link_edge_index_only1)

        # # compute preformence
        # ACC = metrics.accuracy_score(T, Y)
        # AUC = metrics.roc_auc_score(T, S)
        # F1 = metrics.f1_score(T, Y)
        # precision, recall, threshold = metrics.precision_recall_curve(T, S)
        # PR_AUC = metrics.auc(recall, precision)
        # # BACC = balanced_accuracy_score(T, Y)
        # # tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        # # TPR = tp / (tp + fn)
        # PREC = precision_score(T, Y)
        # AP = metrics.average_precision_score(T,S)
        # # KAPPA = cohen_kappa_score(T, Y)
        # RACELL = recall_score(T, Y)

        # print(ACC,AUC,F1,PR_AUC,PREC,AP,RACELL)
        # # 文件路径

        # # 打开文件以追加内容
        # with open(output_file_path, 'a') as file:
        #     # 将指标写入文件
        #     file.write(f"ACC: {ACC}\t")
        #     file.write(f"AUC: {AUC}\t")
        #     file.write(f"F1: {F1}\t")
        #     file.write(f"PR_AUC: {PR_AUC}\t")
        #     file.write(f"PREC: {PREC}\t")
        #     file.write(f"AP: {AP}\t")
        #     file.write(f"RACELL: {RACELL}\n")

    # # 打开文本文件
    # with open(file_AUCs, 'r') as file:
    #     # 读取文件的每一行
    #     lines = file.readlines()
    #
    # # 提取所需列的数据
    # column_data1 = []
    # column_data2 = []
    # for line in lines:
    #     # 根据文件的分隔符（例如空格、逗号等）拆分每一行的数据
    #     values = line.split('\t')  # 如果文件中的列使用制表符分隔，可以使用'\t'作为分隔符
    #     # 假设所需的列是第二列（索引为1），将该列的数据添加到列表中
    #     column_data1.append(values[0])
    #     column_data2.append(values[1])
    #
    # # 绘制曲线图
    # plt.plot(column_data1, column_data2)
    # plt.xlabel('Epoch')
    # plt.ylabel('AUC_dev')
    # plt.title('AUC')
    # plt.savefig(AUC_map)
    # # 显示图形
    # plt.show()
