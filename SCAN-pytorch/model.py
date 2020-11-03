from layers import GraphConvolution, GraphConvolutionSparse, InnerDecoder, Dense
import torch
from torch import dropout, sqrt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import torch.sparse


class SCVA(nn.Module):

    def __init__(self,temperature,hidden1,hidden2,adj,num_features, num_nodes, features_nonzero,num_labels,labels_pos,y_train,device='cpu',one_gcn=True,dropout=0.):
        super(SCVA, self).__init__()

        self.hidden1_dim=hidden1
        self.hidden2_dim=hidden2
        self.adj=adj
        self.temperature=temperature
        self.num_nodes = num_nodes
        self.num_feas = num_features
        self.num_labels = num_labels
        self.features_nonzero = features_nonzero
        self.dropout = dropout
        self.labels_pos = labels_pos
        self.y_train = y_train
        self.one_gcn = one_gcn
        self.device = device
        self.hidden1 = Dense(input_dim = self.num_nodes + self.num_feas,output_dim=hidden1,
                        sparse_inputs=self.one_gcn,dropout=self.dropout)
                             

        self.y_pred_logits = Dense(input_dim = hidden1,output_dim = self.num_labels,
                                    sparse_inputs=False,dropout = self.dropout)
        self.hidden2 = Dense(input_dim=self.num_nodes,output_dim=hidden1,
                             sparse_inputs=True,dropout=self.dropout)      

        self.z_u_mean = Dense(input_dim = hidden1 + self.num_labels,output_dim= hidden2,
                             sparse_inputs=False,dropout=self.dropout)
      
        self.z_u_log_std = Dense(input_dim=hidden1 + self.num_labels,output_dim=hidden2,
                                 sparse_inputs=False,dropout=self.dropout)
        

        self.z_a_mean = Dense(input_dim=hidden1,
                              output_dim=hidden2 + self.num_labels,
                              sparse_inputs=False,dropout=self.dropout)

        self.z_a_log_std = Dense(input_dim = hidden1,output_dim= hidden2 + self.num_labels,
                                 sparse_inputs=False,dropout=self.dropout)
        
        
    

        self.reconstructions = InnerDecoder(input_dim = hidden2 + self.num_labels)

    def gumbel_softmax(self):
        """
        sample from categorical distribution using gumbel softmax trick
        """
        g = -torch.log(-torch.log(torch.Tensor(self.num_nodes,self.num_labels).uniform_()))
        y_pred = torch.exp((torch.log(self.y_pred_prob) + g) / self.temperature)
        y_pred = y_pred / torch.sum(y_pred,1).view(-1,1)   
        return y_pred
    
    
    def reconstruction_y(self):
        """
        get y_pred for reconstruction: replace probabilities of nodes with
        label with label one-hot encoding
        """
        row = self.y_train.size()[0]
        col = self.y_train.size()[1]
        temp_lp=self.labels_pos.repeat(col).reshape(row,col)
        #one hot encoding for label data
        y_pred_reconstruct = torch.where(torch.from_numpy(temp_lp).to(self.device),self.y_train, self.y_pred)
        return y_pred_reconstruct
    
    def yz(self):
        """
        get probability of y to compute z
        """
        row = self.y_train.size()[0]
        col = self.y_train.size()[1]
        temp_lp=self.labels_pos.repeat(col).reshape(row,col)
        # temp_lp=np.tile(self.labels_pos,(len(self.labels_pos),7))
        yz = torch.where(torch.from_numpy(temp_lp).to(self.device),self.y_train, self.y_pred_prob)
        return yz

    def forward(self,Fn,Fa):
        if not self.one_gcn:
            Fn = torch.sparse.mm(self.adj,Fn.to_dense())
        
        h1 = torch.tanh(self.hidden1(Fn))

        #predition of y
        node_fea = torch.sparse.mm(self.adj,h1)
        y_pred_logits = self.y_pred_logits(h1)
        self.y_pred_prob = F.softmax(y_pred_logits,-1)


        h2 = torch.tanh(self.hidden2(Fa.t()))
        
        #embeddings of nodes: mean and log variance
        yz=self.yz() # concat information y

        #use convolution results only
        z_u_mean = self.z_u_mean(torch.cat((node_fea,yz),1))
        z_u_log_std = self.z_u_log_std(torch.cat((node_fea,yz),1))
       
        #embeddings of features
        z_a_mean = self.z_a_mean(h2)
        z_a_log_std = self.z_a_log_std(h2)

        #sampling from embeddings of nodes and features
        z_u = z_u_mean + torch.randn(self.num_nodes, self.hidden2_dim) * torch.exp(z_u_log_std)
        z_a = z_a_mean + torch.randn(self.num_feas, self.hidden2_dim + self.num_labels) * torch.exp(z_a_log_std)
      
        #sampling from y_pred
        self.y_pred = self.gumbel_softmax()

        #get y for reconstruction
        self.y_pred_reconstruction = self.reconstruction_y()

        #concat z_u and y_pred
        zy = torch.cat((z_u,self.y_pred_reconstruction), 1)

        preds_sub_u, preds_sub_a = self.reconstructions(zy, z_a)
        
        return preds_sub_u, preds_sub_a,z_u_mean,z_u_log_std,z_a_mean,z_a_log_std,y_pred_logits,self.y_pred_reconstruction,self.y_pred_prob
    