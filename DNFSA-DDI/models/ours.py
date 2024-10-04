import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv,SAGEConv,GCNConv,GatedGraphConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool 
import pandas as pd
import numpy as np


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=0)
        return (beta * z), beta
        
# Define ISTA-Net-plus Block
class BasicBlock1(nn.Module):
    def __init__(self,pa=881):
        super(BasicBlock1, self).__init__()
        
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))        
        self.fc1 = nn.Linear(pa,512)
        self.fc2 = nn.Linear(pa,pa)
        self.fc3 = nn.Linear(881,256)
        self.fc4 = nn.Linear(881,256)

    def forward(self, x):       
        # x = self.fc1(x_input)
        # x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.soft_thr))# 软阈值操作，保留重要信息
        x = self.fc3(x)
        x = F.relu(x)
        
        # x = self.fc4(x)
        # x = F.relu(x)
        # x_backward = F.dropout(x, p=0.2, training=self.training)

        # x_pred = x_input + x_backward

        return x

class GATNet(torch.nn.Module):
    def __init__(self, micr_feature=128, link_feature =256, hidden_dim =128, outputdim=2, dropout=0.2):
        super(GATNet, self).__init__()
    
        self.attention = Attention(hidden_dim*2)
        # self.ista = ISTANetplus(881)
        self.BasicBlock= BasicBlock1(881)
        # self.cid_mlp1 = Linear(881, 256)


        #GAT#drug1
        self.drug1_gat1 = GATConv(256, hidden_dim,heads=10, dropout=dropout)
        self.drug1_gat2 = GATConv(hidden_dim*10, hidden_dim, dropout=dropout)
        self.drug1_fc_g1 = nn.Linear(hidden_dim, hidden_dim)
        #GAT#net
        # self.drug2_gat1 = GATConv(256, link_feature, heads=10, dropout=dropout)
        # self.drug2_gat2 = GATConv(link_feature*10, hidden_dim, dropout=dropout)
        #GCN#net
        self.drug2_gcn1 = GCNConv(256, link_feature, dropout=dropout)
        self.drug2_gcn2 = GCNConv(link_feature, link_feature, dropout=dropout)
        self.drug2_gcn3 = GCNConv(link_feature, link_feature, dropout=dropout)
        self.drug2_gcn4 = GCNConv(link_feature, hidden_dim, dropout=dropout)
        self.drug2_gcn5 = GCNConv(hidden_dim, hidden_dim, dropout=dropout)

        self.gat11_1 = GATConv(78, 128, dropout=dropout)
        self.gat11_2 = GATConv(128, 128, dropout=dropout)
        self.gat12_1 = GATConv(256, 128, dropout=dropout)
        self.gat12_2 = GATConv(128, 128, dropout=dropout)
        self.gat1 = GATConv(334, 128,heads=10, dropout=dropout)
        self.gat2 = GATConv(1280, 128,dropout=dropout)
        # self.weight1 = nn.Parameter(torch.Tensor([0.4]))  # 初始化权重参数1
        self.weight2 = nn.Parameter(torch.Tensor([0.4]))  # 初始化权重参数2
        self.weight3 = nn.Parameter(torch.Tensor([0.2]))  # 初始化权重参数3
        
        # self.cid_mlp1 = Linear(881, 256)
        # self.cid_mlp2 = Linear(512, 256)
        # self.cid_mlp3 = Linear(512, 78)

        #FC
        self.fc1_1 = Linear(hidden_dim *4, 1024)
        self.fc1_2 = Linear(hidden_dim *2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.out = nn.Linear(64, outputdim)
        
        # activation and regularization
        self.relu = nn.ReLU()
        self.Drop = nn.Dropout()
        
    def forward(self, drug1, drug2, cid, link_edge):
        
        x1, i1, edge_index1, batch1 = drug1.x, drug1.z.long(), drug1.edge_index,drug1.batch
        x2, i2, edge_index2, batch2 = drug2.x, drug2.z.long(), drug2.edge_index,drug2.batch      
        
        #drug chem+substru78+256
        x11 = x1[:, :78]
        x12 = x1[:, 78:]
        x21 = x2[:, :78]
        x22 = x2[:, 78:]
        
        '''# weight1_normalized = self.weight1 / total_weight
        weight2_normalized = self.weight2 
        weight3_normalized = self.weight3''' 
        
        ########################### macro ###########################
        # cid = self.cid_mlp1(cid)
        # cid ,_= self.fingeratt(cid ,cid ,cid )
        # cid = self.trans(cid,cid)
        # cid = self.cid_mlp1(cid)
        
        cid = self.BasicBlock(cid)
                
        link_feature1 = self.drug2_gcn1(cid, link_edge)
        link_feature = F.elu(link_feature1)
        link_feature = F.dropout(link_feature, p=0.2, training=self.training)
        
        link_feature2 = self.drug2_gcn2(link_feature, link_edge)
        link_feature = F.elu(link_feature2)
        link_feature = F.dropout(link_feature, p=0.2, training=self.training)
        
        link_feature2 = self.drug2_gcn3(link_feature, link_edge)
        link_feature = F.elu(link_feature2)
        link_feature = F.dropout(link_feature, p=0.2, training=self.training)
        
        link_feature = self.drug2_gcn4(link_feature, link_edge)
        link_feature = F.elu(link_feature)
        link_feature = F.dropout(link_feature, p=0.2, training=self.training)

        link_feature = self.drug2_gcn5(link_feature, link_edge)
        link_feature = F.elu(link_feature)
        link_feature = F.dropout(link_feature, p=0.2, training=self.training)
        
        link_drug1 = link_feature[i1]
        link_drug2 = link_feature[i2]
        
        ########################### micro ###########################
        # x11 = gmp(x1, batch1)
        
        '''#drug1
        x1 = self.drug1_gat1(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1 = self.drug1_gat2(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
                        
        x1 = global_add_pool(x1, batch1)  # global max pooling
        x1 = self.relu(x1)        

        #drug2
        x2 = self.drug1_gat1(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x2 = self.drug1_gat2(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)          
        
        x2 = global_add_pool(x2, batch2)  # global max pooling
        x2 = self.relu(x2)'''
        
        '''x11 = self.gat11_1(x11,edge_index1)
        x11 = F.elu(x11)
        x11 = F.dropout(x11, p=0.2, training=self.training)
        x11 = self.gat11_2(x11,edge_index1)
        x11 = F.elu(x11)
        x11 = F.dropout(x11, p=0.2, training=self.training)
        
        x12 = self.gat12_1(x12,edge_index1)
        x12 = F.elu(x12)
        x12 = F.dropout(x12, p=0.2, training=self.training)
        x12 = self.gat12_2(x12,edge_index1)
        x12 = F.elu(x12)
        x12 = F.dropout(x12, p=0.2, training=self.training)
        
        x1  = self.gat1(x1,edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1  = self.gat2(x1,edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x1 =  x11 + x12 + x1
        # x1 = torch.mul(x1,x11)
        # x1 = torch.mul(x1,x12)
        # x1 =  x11*weight2_normalized + x12*weight3_normalized
        x1 = global_add_pool(x11, batch1)  # global max pooling
        x1 = self.relu(x1)
        
        x21 = self.gat11_1(x21, edge_index2)
        x21 = F.elu(x21)
        x21 = F.dropout(x21, p=0.2, training=self.training)
        x21 = self.gat11_2(x21, edge_index2)
        x21 = F.elu(x21)
        x21 = F.dropout(x21, p=0.2, training=self.training)

        x22 = self.gat12_1(x22, edge_index2)
        x22 = F.elu(x22)
        x22 = F.dropout(x22, p=0.2, training=self.training)
        x22 = self.gat12_2(x22, edge_index2)
        x22 = F.elu(x22)
        x22 = F.dropout(x22, p=0.2, training=self.training)

        x2 = self.gat1(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = self.gat2(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x2 =  x21 + x22 +x2
        # x2 =  x21*weight2_normalized + x22*weight3_normalized
        # x2 = torch.mul(x21,x22)
        # x2 = torch.mul(x2,x21)
        # x2 = torch.mul(x2,x22)
        x2 = global_add_pool(x21, batch2)  # global max pooling
        x2 = self.relu(x2)'''
        ########################### fusion ###########################        
        
        '''drug_1 = torch.cat((x1, link_drug1), 1)
        drug_2 = torch.cat((x2, link_drug2), 1)
        drug_1, _  = self.attention(drug_1)
        drug_2, _  = self.attention(drug_2)

        xc = torch.cat((drug_1, drug_2), 1)'''
        
        '''xc = torch.cat((x1, link_drug1, x2, link_drug2), 1)        
        xc, _  = self.attention(xc)'''
        
        # drug_1 = drug_1.unsqueeze(1).T
        # drug_2 = drug_2.unsqueeze(1).T       
               
        # xc = torch.cat((x1, link_drug1, x2, link_drug2), 1)
        # xc = torch.cat((x1, x2), 1)
        xc = torch.cat((link_drug1, link_drug2), 1)
        xc = F.normalize(xc, 2, 1)
        #FC
        xc = self.fc1_2(xc)
        xc = self.relu(xc)
        xc = self.Drop(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.Drop(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.Drop(xc)
        out = self.out(xc)
        return out