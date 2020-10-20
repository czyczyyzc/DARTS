import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def complementary_transition_matrix(ncls):
    rho = 1.0
    M = (rho / (ncls - 1)) * np.ones((ncls, ncls))  #
    for i in range(ncls):
        M[i, i] = 1. - rho
    M = torch.from_numpy(M).float().cuda()
    return M


# Q10 = complementary_transition_matrix(10)


def forward_loss(x, target, Q):
    # Q = (Q + 1e-12) / (1 + Q.size()[0] * 1e-12)
    prob = F.softmax(x, dim=1)
    prob = torch.mm(prob, Q)
    out  = torch.log(prob)
    loss = F.nll_loss(out, target)
    return loss


def forward_loss_co(x, target, Q):
    probt = F.softmax(x,dim=1)
    prob = torch.mm(probt, Q.cuda())
    out = torch.log(prob)
    loss = F.nll_loss(out, target, reduction='none')
    return loss


def class100_loss(x,target):
    probt = F.softmax(x,dim=1)
    Q100 = complementary_transition_matrix(100)
    prob = torch.mm(probt, Q100)
    out = -(torch.log(prob))
    sum = 0
    for k in range(len(x)):
        kk = out[k,:]
        tt = target[k,:]
        
        sum += torch.sum(kk[tt.tolist()])
    loss = sum / len(x)/x.size()[1]
    #print(loss)
    return loss


def BCEloss(x,target): 
    #x = nn.Sigmoid(x,dim=1)  
    probt = F.softmax(x,dim=1)
    Q100 = complementary_transition_matrix(100) 
    prob = torch.mm(probt, Q100) 
    one_hot_label = torch.zeros(x.size()).scatter_(1,target.cpu(),1)
    #output = nn.BCELoss(prob, target)
    loss = nn.BCEWithLogitsLoss()
    output = loss(prob, one_hot_label.cuda())
    return output

