import numpy as np
from random import randint
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from geomloss import SamplesLoss
from sklearn.model_selection import train_test_split
import sklearn.metrics
import random 

def index_gen(state, test_s, grade):
    a = [i for i in range(len(grade))]
    a_1, a_2, a_3, a_4 = train_test_split(a, grade, test_size=test_s, random_state = state,  stratify = grade)
    index_train = np.array(a_1)
    index_test = np.array(a_2)
    return index_train, index_test


class conv_ls_w(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, rep_dim, num_class):
        super(conv_ls_w, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 300, out_channels = 300, kernel_size=1,padding=0)
        self.t0 = nn.Parameter(torch.rand((num_class, rep_dim, 300), requires_grad=True))

    def forward(self, sentence_embd):
        res = self.conv1(sentence_embd.permute(0, 2, 1)).permute(0, 2, 1)

        return res
        
def KNN_pred_test_ws(model, look_up_test, length_test, grade_test, weight_test, num_class):
    embd_test = model(look_up_test.cuda())
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    weight_ref = torch.ones(50)/50
    weight_ref = weight_ref.cuda()

    dis = np.zeros((len(embd_test), num_class))
    for i in range(len(embd_test)):
        for j in range(num_class):
            dis[i][j] = loss(weight_test[i][0:length_test[i]], embd_test[i][0:length_test[i]], weight_ref, model.t0[j])


    acc = 0
    for i in range(len(embd_test)):
        pred = np.argsort(dis[i])[0]
        #print(dis[i], pred, grade_test[i])
        if pred == grade_test[i]:
            acc = acc + 1

    return acc/len(grade_test)

class Loss_reference(nn.Module):
    def __init__(self, flag):
        super(Loss_reference, self).__init__()
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        self.flag = flag

    
    def forward(self, anchor, length_anchor, grade, weight, model, num_class) -> torch.Tensor:
        loss = torch.zeros(len(anchor))
        post = torch.zeros(len(anchor))
        negt = torch.zeros(len(anchor)*(num_class-1))
        weight_ref = torch.ones(50)/50
        weight_ref = weight_ref.cuda()
        #dis = self.loss(model.t0, model.t1) + self.loss(model.t0, model.t2) + self.loss(model.t0, model.t3) + self.loss(model.t0, model.t4) + self.loss(model.t1, model.t2) + self.loss(model.t1, model.t3) + self.loss(model.t1, model.t4) + self.loss(model.t2, model.t3) + self.loss(model.t2, model.t4) + self.loss(model.t3, model.t4)
        dis = 0
        
        for i in range(num_class):
            for j in range(num_class):
                dis = dis + self.loss(model.t0[i], model.t0[j])
        
        for i in range(len(anchor)):
            distance_positive = self.loss(weight[i][0:length_anchor[i]], anchor[i][0:length_anchor[i]], weight_ref, model.t0[int(grade[i])])
            distance_negative = torch.zeros(num_class-1)
            ind = 0
            post[i] = distance_positive
            for k in range(num_class):
                if k != int(grade[i]):
                    distance_negative[ind] = self.loss(weight[i][0:length_anchor[i]], anchor[i][0:length_anchor[i]], weight_ref, model.t0[k])
                    ind = ind + 1
            
            losses = 0
            for j in range(num_class-1):
                losses = losses + torch.relu(distance_positive - distance_negative[j] + 10)

            loss[i] = losses
            post[i] = distance_positive
            negt[(num_class-1)*i:(num_class-1)*i+num_class-1] = distance_negative
            #print(distance_negative)

        if self.flag == 0:
            return loss.mean()
        else:
            return loss.mean() - dis/100
        
