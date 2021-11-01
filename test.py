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

def node_pair(grade, num):
    test = np.zeros((len(grade)*num, 3)).astype(int)
    p = 0
    for i in range(len(grade)):
        for j in range(num):
            test[p][0] = i
            pos = randint(0, len(grade_sort))
            neg = randint(0, len(grade_sort))
            while grade[pos] != grade[i]:
                pos = randint(0, len(grade_sort))
            while grade[neg] == grade[i]:
                neg = randint(0, len(grade_sort))
            test[p][1] = pos
            test[p][2] = neg
                
    li=[i for i in range(len(test))]
    random.shuffle(li) 
    test = test[li]
    test = test.reshape(int(len(test)/440), 440, 3) 
    return test
     
'''
class Loss_reference(nn.Module):
    def __init__(self):
        super(Loss_reference, self).__init__()
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    
    def forward(self, anchor, length_anchor, grade, model) -> torch.Tensor:
        loss = torch.zeros(len(anchor))
        post = torch.zeros(len(anchor))
        negt = torch.zeros(len(anchor)*4)
        dis = self.loss(model.t0, model.t1) + self.loss(model.t0, model.t2) + self.loss(model.t0, model.t3) + self.loss(model.t0, model.t4) + self.loss(model.t1, model.t2) + self.loss(model.t1, model.t3) + self.loss(model.t1, model.t4) + self.loss(model.t2, model.t3) + self.loss(model.t2, model.t4) + self.loss(model.t3, model.t4)
        for i in range(len(anchor)):
            if grade[i] == 0:
                distance_positive = self.loss(anchor[i][0:length_anchor[i]], model.t0)
                distance_negative_1 = self.loss(anchor[i][0:length_anchor[i]], model.t1)
                distance_negative_2 = self.loss(anchor[i][0:length_anchor[i]], model.t2)
                distance_negative_3 = self.loss(anchor[i][0:length_anchor[i]], model.t3)
                distance_negative_4 = self.loss(anchor[i][0:length_anchor[i]], model.t4)
            if grade[i] == 1:
                distance_positive = self.loss(anchor[i][0:length_anchor[i]], model.t1)
                distance_negative_1 = self.loss(anchor[i][0:length_anchor[i]], model.t0)
                distance_negative_2 = self.loss(anchor[i][0:length_anchor[i]], model.t2)
                distance_negative_3 = self.loss(anchor[i][0:length_anchor[i]], model.t3)
                distance_negative_4 = self.loss(anchor[i][0:length_anchor[i]], model.t4)
            if grade[i] == 2:
                distance_positive = self.loss(anchor[i][0:length_anchor[i]], model.t2)
                distance_negative_1 = self.loss(anchor[i][0:length_anchor[i]], model.t1)
                distance_negative_2 = self.loss(anchor[i][0:length_anchor[i]], model.t0)
                distance_negative_3 = self.loss(anchor[i][0:length_anchor[i]], model.t3)
                distance_negative_4 = self.loss(anchor[i][0:length_anchor[i]], model.t4)
            if grade[i] == 3:
                distance_positive = self.loss(anchor[i][0:length_anchor[i]], model.t3)
                distance_negative_1 = self.loss(anchor[i][0:length_anchor[i]], model.t1)
                distance_negative_2 = self.loss(anchor[i][0:length_anchor[i]], model.t2)
                distance_negative_3 = self.loss(anchor[i][0:length_anchor[i]], model.t0) 
                distance_negative_4 = self.loss(anchor[i][0:length_anchor[i]], model.t4) 
            if grade[i] == 4:
                distance_positive = self.loss(anchor[i][0:length_anchor[i]], model.t4)
                distance_negative_1 = self.loss(anchor[i][0:length_anchor[i]], model.t1)
                distance_negative_2 = self.loss(anchor[i][0:length_anchor[i]], model.t2)
                distance_negative_3 = self.loss(anchor[i][0:length_anchor[i]], model.t0) 
                distance_negative_4 = self.loss(anchor[i][0:length_anchor[i]], model.t3) 
            #losses = torch.relu(distance_positive - distance_negative_1 + 10) + torch.relu(distance_positive - distance_negative_2 + 10) + torch.relu(distance_positive - distance_negative_3 + 10) + torch.relu(distance_positive - distance_negative_4 + 10)
            #print(int(post), int(distance_negative_1), int(distance_negative_2), int(distance_negative_3)
            losses = - torch.log(torch.exp(-distance_positive/20) / (torch.exp(-distance_positive/20) + torch.exp(-distance_negative_1/20) + torch.exp(-distance_negative_2/20) + torch.exp(-distance_negative_3/20) + torch.exp(-distance_negative_4/20)))
            loss[i] = losses
            post[i] = post
            negt[4*i] = distance_negative_1
            negt[4*i+1] = distance_negative_2
            negt[4*i+2] = distance_negative_3
            negt[4*i+3] = distance_negative_4
        print(int(distance_positive), int(distance_negative_1), int(distance_negative_2), int(distance_negative_3))

        return loss.mean() - dis/40 + post.mean()/40 + negt.mean()/40
'''      
def dis():
    model = torch.load('model.pkl')
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    length = np.load('BBCsports_length.npy')
    grade = np.load("BBCsports_grade.npy")[0].round()-1
    look_up = np.load("BBCsports.npy")
    look_up = torch.from_numpy(look_up).float()
    rep = model(look_up.cuda(), torch.from_numpy(length).float().cuda())
    dis = np.zeros((742, 742))
    for i in range(737):
        print(i)
        for j in range(i):
            dis[i][j] = loss(rep[i][0:length[i]], rep[j][0:length[j]]).cpu().detach().numpy()
            
    for i in range(737):
        dis[737][i] = loss(rep[i][0:length[i]], model.t0).cpu().detach().numpy()
        dis[738][i] = loss(rep[i][0:length[i]], model.t1).cpu().detach().numpy()
        dis[739][i] = loss(rep[i][0:length[i]], model.t2).cpu().detach().numpy()
        dis[740][i] = loss(rep[i][0:length[i]], model.t3).cpu().detach().numpy()
        dis[741][i] = loss(rep[i][0:length[i]], model.t4).cpu().detach().numpy()
    
    #dis_np = dis.cpu().detach().numpy()
    np.save('dis.npy', dis)
    return dis
    

class conv_ls_w(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, rep_dim, num_class):
        super(conv_ls_w, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 300, out_channels = 300, kernel_size=1,padding=0)
        self.hidden_dim = hidden_dim
        self.bn1 = nn.BatchNorm1d(300)
        self.hidden2tag = nn.Parameter(torch.rand(embedding_dim, requires_grad=True))
        self.t0 = nn.Parameter(torch.rand((num_class, rep_dim, 300), requires_grad=True))

    def forward(self, sentence_embd):
        '''
        #embeds = self.word_embeddings(sentence)
        temp = torch.mul(self.hidden2tag, sentence_embd)
        s_1, s_2, s_3 = temp.shape
        d = temp.sum(2)
        temp_2 = torch.div(temp.view(s_1*s_2, 300).t(), d.view(s_1*s_2)+0.00000001).t()
        #print(temp_2.shape, s_1*s_2)
        temp_3 = temp_2.reshape(s_1, s_2, s_3)
        #print(temp_2.view(s_1, s_2, s_3).shape, sentence_embd.shape)
        res = self.conv1(temp_3.permute(0, 2, 1)).permute(0, 2, 1)
        '''
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
        
        
class Loss_reference_neg(nn.Module):
    def __init__(self):
        super(Loss_reference_neg, self).__init__()
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    
    def forward(self, anchor, length_anchor, grade, weight, anchor_out_neg, model, num_class) -> torch.Tensor:
        loss = torch.zeros(len(anchor))
        post = torch.zeros(len(anchor))
        negt = torch.zeros(len(anchor)*(num_class))
        weight_neg = torch.ones(100)/100
        weight_neg = weight_neg.cuda()
        weight_ref = torch.ones(50)/50
        weight_ref = weight_ref.cuda()
        #dis = self.loss(model.t0, model.t1) + self.loss(model.t0, model.t2) + self.loss(model.t0, model.t3) + self.loss(model.t0, model.t4) + self.loss(model.t1, model.t2) + self.loss(model.t1, model.t3) + self.loss(model.t1, model.t4) + self.loss(model.t2, model.t3) + self.loss(model.t2, model.t4) + self.loss(model.t3, model.t4)
        dis = 0
        
        for i in range(len(anchor)):
            distance_positive = self.loss(weight[i][0:length_anchor[i]], anchor[i][0:length_anchor[i]], weight_ref, model.t0[int(grade[i])])
            distance_negative = torch.zeros(num_class)
            ind = 0
            post[i] = distance_positive
            for k in range(num_class):
                if k != int(grade[i]):
                    distance_negative[ind] = self.loss(weight[i][0:length_anchor[i]], anchor[i][0:length_anchor[i]], weight_ref, model.t0[k])
                    ind = ind + 1
            distance_negative[-1] = self.loss(weight_ref, model.t0[k], weight_neg, anchor_out_neg[i])
            losses = 0
            for j in range(num_class-1):
                losses = losses + torch.relu(distance_positive - distance_negative[j] + 10)
                
            #losses = losses + torch.relu(distance_positive - distance_negative[-1] + 5)

            loss[i] = losses/5
            post[i] = distance_positive
            negt[(num_class)*i:(num_class)*i+num_class] = distance_negative

        print(int(post.mean()), int(negt.mean()))

        return loss.mean()
        
class Loss_reference_neg_2(nn.Module):
    def __init__(self):
        super(Loss_reference_neg_2, self).__init__()
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    
    def forward(self, anchor, length_anchor, grade, weight, anchor_out_neg, grade_neg, model, num_class) -> torch.Tensor:
        loss = torch.zeros(len(anchor))
        post = torch.zeros(len(anchor))
        negt = torch.zeros(len(anchor)*(num_class))
        weight_neg = torch.ones(100)/100
        weight_neg = weight_neg.cuda()
        weight_ref = torch.ones(50)/50
        weight_ref = weight_ref.cuda()
        #dis = self.loss(model.t0, model.t1) + self.loss(model.t0, model.t2) + self.loss(model.t0, model.t3) + self.loss(model.t0, model.t4) + self.loss(model.t1, model.t2) + self.loss(model.t1, model.t3) + self.loss(model.t1, model.t4) + self.loss(model.t2, model.t3) + self.loss(model.t2, model.t4) + self.loss(model.t3, model.t4)
        dis = 0
        
        for i in range(len(anchor)):
            distance_positive = self.loss(weight[i][0:length_anchor[i]], anchor[i][0:length_anchor[i]], weight_ref, model.t0[int(grade[i])])
            distance_negative = torch.zeros(num_class)
            ind = 0
            post[i] = distance_positive
            for k in range(num_class):
                if k != int(grade[i]):
                    distance_negative[ind] = self.loss(weight[i][0:length_anchor[i]], anchor[i][0:length_anchor[i]], weight_ref, model.t0[k])
                    ind = ind + 1
            distance_negative[-1] = self.loss(weight_ref, model.t0[k], weight_neg, anchor_out_neg[i])
            losses = 0
            for j in range(num_class):
                losses = losses + torch.relu(distance_positive - distance_negative[j] + 10)
                
            loss_dif = torch.abs(self.loss(weight_ref, model.t0[grade_neg[i]], weight_neg, anchor_out_neg[i]) - self.loss(weight_ref, model.t0[k], weight_neg, anchor_out_neg[i]))
            losses = losses + loss_dif
            #losses = losses + torch.relu(distance_positive - distance_negative[-1] + 5)

            loss[i] = losses
            post[i] = distance_positive
            negt[(num_class)*i:(num_class)*i+num_class] = distance_negative

        print(int(post.mean()), int(negt.mean()))

        return loss.mean() 
 