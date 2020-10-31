from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
import time
import os
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import argparse
# from optimizer import  OptimizerSCVA
from input_data import load_AN
from model import SCVA
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_feas,get_labels,labels_onehot
from classification import embedding_classifier

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=128,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=64,
                    help='Number of units in hidden layer 2.')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--ratio', type=float, default=0.1,
                    help='ratio.')
parser.add_argument('--temperature', type=float, default=0.2,
                    help='temperature of the gumbel softmax.')
parser.add_argument('--lamb', type=float, default=270,
                    help='lambda.')
parser.add_argument('--beta', type=float, default=0.1,
                    help='beta.')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='alpha.')
parser.add_argument('--dataset', type=str, default="BlogCatalog",
                    help='Dataset string.')
args = parser.parse_args()
dataset_str = args.dataset
weight_decay = args.weight_decay


link_predic_result_file = "result/AGAE_{}.res".format(dataset_str)
embedding_node_mean_result_file = "result/AGAE_{}_n_mu.emb".format(dataset_str)
embedding_attr_mean_result_file = "result/AGAE_{}_a_mu.emb".format(dataset_str)
embedding_node_var_result_file = "result/AGAE_{}_n_sig.emb".format(dataset_str)
embedding_attr_var_result_file = "result/AGAE_{}_a_sig.emb".format(dataset_str)


"load data"
adj, features = load_AN(dataset_str)
labels_pos,labels_test_pos,labels,full_labels,num_labels = get_labels(dataset_str,ratio=args.ratio)
#all_labels_onehot: tensor, one-hot encoding of the labels
y_train = labels_onehot(labels_pos,labels,num_labels)

adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis,:],[0]),shape=adj.shape)
adj.eliminate_zeros()

num_nodes = adj.shape[0]
features_coo = sparse_to_tuple(features.tocoo())
num_features = features_coo[2][1]
features_nonzero = features_coo[1].shape[0]
one_gcn = False if dataset_str == "cora" else True


#split the dataset into training, validation and test set
#adj_train and fea_train is the sparse matrix used for training
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features)

adj_orig = adj = adj_train
features_orig = fea_train
features =  sparse_to_tuple(fea_train.tocoo())

#concate adjacant matrix and features to get Fn
Fn_train = sparse_to_tuple(sp.hstack((adj_train,fea_train)))
Fn_train=torch.sparse.FloatTensor(torch.LongTensor(Fn_train[0].astype(np.int64)).t(),torch.FloatTensor(Fn_train[1]),Fn_train[2]) 
Fa_train = sparse_to_tuple(fea_train.tocoo())
Fa_train=torch.sparse.FloatTensor(torch.LongTensor(Fa_train[0].astype(np.int64)).t(),torch.FloatTensor(Fa_train[1]),Fa_train[2])        


# Create model
adj_train_mat =  preprocess_graph(adj_train)
adj_train_mat=torch.sparse.FloatTensor(torch.LongTensor(adj_train_mat[0].astype(np.int64)).t(),torch.FloatTensor(adj_train_mat[1]),adj_train_mat[2]) 

y_train = y_train.float()

model = SCVA(args.temperature,args.hidden1,args.hidden2,adj_train_mat,num_features, num_nodes, features_nonzero,num_labels,labels_pos,y_train,one_gcn)
pos_weight_u = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm_u = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
pos_weight_a = float(features[2][0] * features[2][1] - len(features[1])) / len(features[1])
norm_a = features[2][0] * features[2][1] / float((features[2][0] * features[2][1] - len(features[1])) * 2)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)



def get_roc_score(edges_pos, edges_neg,preds_sub_u):
    
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    # adj_rec = sess.run(model.reconstructions[0], feed_dict=feed_dict).reshape([num_nodes, num_nodes])
    adj_rec=preds_sub_u.view(num_nodes,num_nodes).data.numpy()
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    # print(np.min(adj_rec))
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg]).astype(np.float)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score_a(feas_pos, feas_neg,preds_sub_a):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    # fea_rec = sess.run(model.reconstructions[1], feed_dict=feed_dict).reshape([num_nodes, num_features])
    fea_rec = preds_sub_a.view(num_nodes, num_features).data.numpy()
    preds = []
    pos = []
    for e in feas_pos:
        preds.append(sigmoid(fea_rec[e[0], e[1]]))
        pos.append(features_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in feas_neg:
        preds_neg.append(sigmoid(fea_rec[e[0], e[1]]))
        neg.append(features_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def weighted_cross_entropy_with_logits(logits, targets, pos_weight):
    logits=logits.clamp(-10,10)
    return targets * -torch.log(torch.sigmoid(logits)) *pos_weight + (1 - targets) * -torch.log(1 - torch.sigmoid(logits))

def softmax_cross_entropy_with_logits_v2(logits,targets):
    logsoftmax = nn.LogSoftmax(dim=-1)
    loss=torch.sum(-targets * logsoftmax(logits), 1)
    return loss

cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_label=torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].astype(np.int64)).t(),torch.FloatTensor(adj_label[1]),adj_label[2]) 
features_label = sparse_to_tuple(fea_train)
features_label=torch.sparse.FloatTensor(torch.LongTensor(features_label[0].astype(np.int64)).t(),torch.FloatTensor(features_label[1]),features_label[2]) 

# Train model

for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    preds_sub_u, preds_sub_a,z_u_mean,z_u_log_std,z_a_mean,z_a_log_std,y_pred_logits,y_pred_reconstruction,y_pred_prob=model(Fn_train,Fa_train)
    labels_sub_u=torch.from_numpy(adj_orig.toarray()).flatten().float()
    labels_sub_a=torch.from_numpy(features_orig.toarray()).flatten().float()

    # compute reconstruction loss
    cost_u = norm_u * torch.mean(weighted_cross_entropy_with_logits(logits=preds_sub_u, targets=labels_sub_u, pos_weight=pos_weight_u))
    cost_a = norm_a * torch.mean(weighted_cross_entropy_with_logits(logits=preds_sub_a, targets=labels_sub_a, pos_weight=pos_weight_a))
    cost_recon = args.alpha*cost_u + (1-args.alpha)*cost_a

    # compute kl divergence
    kl_u = args.alpha*(0.5 ) * torch.mean((1 + 2 * z_u_log_std - z_u_mean.pow(2) - (torch.exp(2*z_u_log_std))).sum(1))/ num_nodes
    kl_a = (1-args.alpha)*(0.5 ) * torch.mean((1 + 2 * z_a_log_std - (z_a_mean.pow(2)) -(torch.exp(2*z_a_log_std))).sum(1))/ num_features
    kl = args.beta * ((num_nodes + num_features) *3* kl_u  / num_nodes + num_nodes / num_features * kl_a) 

    # compute P(y) of labelled nodes
    labelled_pos = labels_pos
    num_labelled_nodes = np.sum(labelled_pos)
    num_unlabelled_nodes = len(labelled_pos) - num_labelled_nodes
    unlabelled_pos = np.logical_not(labelled_pos)
    labelled_pos=torch.from_numpy(labelled_pos).to(torch.uint8)==True
    unlabelled_pos=torch.from_numpy(unlabelled_pos).to(torch.uint8)==True
    labelled_y_pred_logits =  y_pred_logits[labelled_pos]
    labelled_y_true = y_train[labelled_pos]
    unlabelled_y_pred_logits = y_pred_logits[unlabelled_pos]
    unlabelled_y_true = y_pred_reconstruction[unlabelled_pos]#4677 6

    label_entropy =  args.lamb*softmax_cross_entropy_with_logits_v2(labelled_y_pred_logits,labelled_y_true) 
    unlabel_entropy = softmax_cross_entropy_with_logits_v2(unlabelled_y_pred_logits,unlabelled_y_true)

    two_label_py = torch.sum(label_entropy.view(1, -1).repeat(num_labelled_nodes, 1)+label_entropy.view(-1, 1))
    two_unlabel_py = torch.sum(unlabel_entropy.view(1, -1).repeat(num_unlabelled_nodes, 1)+unlabel_entropy.view(-1, 1))
    one_label_py = 2 * torch.sum(unlabel_entropy.view(1, -1).repeat(num_labelled_nodes, 1)+label_entropy.view(-1, 1))
    
    nodes_py = (two_label_py + two_unlabel_py + one_label_py) / (num_nodes * num_nodes)
    features_py = (torch.sum(label_entropy) + torch.sum(unlabel_entropy)) / (num_nodes)
    
    py = args.alpha*nodes_py + (1-args.alpha)*features_py
    
    # compute shannoy entropy of P(y) of unlabelled nodes
    unlabelled_y_pred_prob = y_pred_prob[unlabelled_pos]
    entropy_unlabelled_y = -torch.sum(unlabelled_y_pred_prob * torch.log(unlabelled_y_pred_prob), 1)
    two_unlabel_entropy_y = torch.sum(entropy_unlabelled_y.view(1, -1).repeat(num_unlabelled_nodes, 1)+entropy_unlabelled_y.view(-1, 1))
    one_unlabel_entropy_y = 2 * torch.sum(entropy_unlabelled_y.view(1, -1).repeat(num_labelled_nodes, 1))
    entropy_y = (two_unlabel_entropy_y + one_unlabel_entropy_y) / ((num_labelled_nodes + num_unlabelled_nodes) * num_unlabelled_nodes)
    entropy_y = args.alpha*entropy_y + (1-args.alpha)*torch.mean(entropy_unlabelled_y)  # add entropy of unlabel nodes for attributes
    
    """final cost"""
    cost = cost_recon - kl + py + entropy_y

    correct_prediction_u = torch.sum(torch.eq(torch.ge(torch.sigmoid(preds_sub_u), 0.5).float(),labels_sub_u).float())/len(labels_sub_u)
    correct_prediction_a = torch.sum(torch.eq(torch.ge(torch.sigmoid(preds_sub_a), 0.5).float(),labels_sub_a).float())/len(labels_sub_a)
    accuracy = torch.mean(correct_prediction_u + correct_prediction_a)
    # Compute average loss
    avg_cost = cost
    avg_accuracy = accuracy

    ## Run single weight update
    avg_cost.backward()
    optimizer.step()
    # Get score
    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false,preds_sub_u)
    roc_curr_a, ap_curr_a = get_roc_score_a(val_feas, val_feas_false,preds_sub_a)
    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(avg_cost),
              "log_lik=", "{:.5f}".format(cost_recon),
              "KL=", "{:.5f}".format(kl),
              "py=","{:.5f}".format(py),
              "cost_a=","{:.5f}".format(cost_a),
              "entropy_y=","{:.5f}".format(entropy_y),
              "train_acc=", "{:.5f}".format(avg_accuracy),
              "val_edge_roc=", "{:.5f}".format(val_roc_score[-1]),
              "val_edge_ap=", "{:.5f}".format(ap_curr),
              "val_attr_roc=", "{:.5f}".format(roc_curr_a),
              "val_attr_ap=", "{:.5f}".format(ap_curr_a),
              "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")    

preds_sub_u, preds_sub_a,z_u_mean,z_u_log_std,z_a_mean,z_a_log_std,y_pred_logits,y_pred_reconstruction,y_pred_prob=model(Fn_train,Fa_train)
roc_score, ap_score = get_roc_score(test_edges, test_edges_false,preds_sub_u)
roc_score_a, ap_score_a = get_roc_score_a(test_feas, test_feas_false,preds_sub_a)



np.save(embedding_node_mean_result_file, z_u_mean.data.numpy())
np.save(embedding_attr_mean_result_file, z_a_mean.data.numpy())

np.save(embedding_node_var_result_file, z_u_log_std.data.numpy())
np.save(embedding_attr_var_result_file, z_a_log_std.data.numpy())    
print('Test edge ROC score: ' + str(roc_score))
print('Test edge AP score: ' + str(ap_score))
print('Test attr ROC score: ' + str(roc_score_a))
print('Test attr AP score: ' + str(ap_score_a))

#get classification accuracy of Discriminator
y_pred_train = np.argmax(y_pred_prob.data.numpy(),axis=1)+1
unlabel_pos = np.logical_not(labels_pos)
acc_train = np.mean(y_pred_train[unlabel_pos] == full_labels[unlabel_pos])
print("ACC of SCVA_DIS:{:.4f}".format(acc_train))

macro_f1_avg,micro_f1_avg,accuracy = embedding_classifier(dataset_str)
print("ACC of SCVA_SVM:" + str(accuracy))
