import csv
from itertools import islice

from torch_geometric.nn import Node2Vec
from sklearn.decomposition import PCA
from torch_geometric.nn.models import node2vec
import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils_test import *
import pickle

device = torch.device('cuda:1')


# def get_cell_feature(cellId, cell_features):    #生成细胞系的特征
#     for row in islice(cell_features, 0, None):
#         if row[0] == cellId:
#             return row[1: ]

# 产生原子的特征
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])  # 这两个函数用于将原子(atom)的不同特征（如符号、度数、氢原子数、隐式价、是否芳香性）进行独热编码，以生成原子的特征向量。


'''其中对于原子的符号、度数、氢原子数和隐式价均使用了长度为11的独热编码，对于是否为芳香性则用了一个长度为1的二元独热编码。
在原子特征向量的末尾，还加上了一个二元特征来指示原子是否为芳香性，因此总共有45+2=47个特征。'''


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):  # 将smile串转换成图
    mol = Chem.MolFromSmiles(smile)  # 从SMILES字符串中创建Molecule对象
    c_size = mol.GetNumAtoms()  # 获取Molecule中的原子数目

    features = []
    for atom in mol.GetAtoms():  # 遍历Molecule中的每个原子
        feature = atom_features(atom)  # 提取原子特征，并将其归一化
        features.append(feature / sum(feature))
    features = np.array(features)

    edges = []
    for bond in mol.GetBonds():  # 遍历Molecule中的每个键
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])  # 添加键的起始原子索引和结束原子索引作为边的一部分
    G = nx.Graph(edges).to_directed()  # 使用边列表创建有向图
    edge_index = []

    for e1, e2 in G.edges:  # 遍历有向图中的每条边
        edge_index.append([e1, e2])

    edge_index_Tensor = torch.tensor(edge_index).T

    # node2vec
    node2vec = Node2Vec(edge_index_Tensor, embedding_dim=256, walk_length=8, context_size=4, num_nodes=c_size)
    features_s = node2vec.embedding.weight
    features_s_list = [np.array(x) for x in features_s.tolist()]

    # features = normalize(features)
    features = np.concatenate([features, features_s_list], axis=1)

    # 返回原子数目、原子特征和边索引##邻接矩阵
    # return c_size, features, edge_index ,edge_index_s
    # return c_size,len(edge_index),edges
    return c_size, features, edge_index  # 不包括RDKit


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def creat_data(datafile):
    '''compound_iso_smiles = []
    df = pd.read_csv('data/DDI_613_SMILE.csv')
    compound_iso_smiles += list(df['smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    print('compound_iso_smiles', compound_iso_smiles)
    for smile in compound_iso_smiles:
        print('smiles', smile)
        g = smile_to_graph(smile)
        smile_graph[smile] = g'''
    # compound_iso_smiles = []  # 建立compound_iso_smiles
    # df = pd.read_csv('data/smilesS.csv')  #  读文件
    # compound_iso_smiles += list(df['smile'])  # 将smile串加进来
    # compound_iso_smiles = set(compound_iso_smiles)  # 变成集合
    # 然后遍历集合中的smile串
    smile_graph = {}  # 建空字典
    # print('compound_iso_smiles', compound_iso_smiles) #只剩下药物对应的simles
    # with open('data/filename.txt', 'w') as f:
    with open('data/STNN/DrugID_DrugCID_DrugName.csv', 'r') as file:
        reader = csv.reader(file)
        data = list(zip(*reader))  # 转置操作，将行转换为列
        for i in range(1, len(data[0])):
            name = data[2][i]
            smile = data[3][i]
            print(name, smile)
            g = smile_to_graph(smile)  # 将SMILES串转换为图形结构 #g = 这个smile串对应的原子数目，特征数目和边的索引
            smile_graph[name] = (i - 1, g)  # simle串 和 g 是字典的键和值  将化学键转为具体的数字，代表特征向量 # 将SMILES串和其对应的图形结构加入到字典中
    # f.write(f"{smile}\t")
    # f.write(f" {g}\n")

    datasets = datafile  # 将datafile转化为datasets
    # convert to PyTorch data format
    processed_data_file_train = 'data/processed/' + datasets + '.pt'

    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv('data/STNN/' + datasets + '.csv')
        # df = pd.read_csv('data/' + datasets + '.csv')
        drug1, drug2, label = list(df['drug1']), list(df['drug2']), list(df['label'])
        drug1, drug2, label = np.asarray(drug1), np.asarray(drug2), np.asarray(label)
        # make data PyTorch Geometric

        print('开始创建数据')
        # TestbedDataset(root='data', dataset=datafile + '_drug1_both_noskip7878', xd=drug1, y=label,smile_graph=smile_graph)
        # TestbedDataset(root='data', dataset=datafile + '_drug2_both_noskip7878', xd=drug2, y=label,smile_graph=smile_graph)

        TestbedDataset(root='data', dataset=datafile + '_drug1_STNN78256', xd=drug1, y=label, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset=datafile + '_drug2_STNN78256', xd=drug2, y=label, smile_graph=smile_graph)

        print('创建数据成功')
        print('preparing ', datasets + '_.pt in pytorch format!')

        print(processed_data_file_train, ' have been created')
    else:
        print(processed_data_file_train, ' are already created')


if __name__ == "__main__":
    # datafile = 'prostate'
    # da = ['DDI_612']
    da = ['STNN01']
    # da = ['test_S1']
    for datafile in da:  # 遍历每一行
        creat_data(datafile)

