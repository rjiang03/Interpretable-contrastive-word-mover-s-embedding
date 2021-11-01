import numpy as np
import sklearn.metrics
from geomloss import SamplesLoss
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 

from test import index_gen
from test import conv_ls_w
from test import Loss_reference
from test import KNN_pred_test_ws
import sklearn.metrics



parser = argparse.ArgumentParser()

parser.add_argument('--lr', default='0.1',
                    help="learning rate")
parser.add_argument('--number', default=50,
                    help="number of reference points")
parser.add_argument('--seed', default=1,
                    help="0-4, predefined train_test_split")

def DataLoader():
    length = np.load('BBCsports_length.npy')
    grade = np.load("BBCsports_grade.npy")[0].round()-1
    look_up = np.load("BBCsports.npy")
    weight = np.load('weight.npy')
    look_up = torch.from_numpy(look_up).float()
    weight = torch.from_numpy(weight).float()
    return look_up, grade, length, weight


def train(model, args, look_up, grade, length, weight, num_class):
    epoch_num = 50
    acc = np.zeros((epoch_num+1, 4))
    for j in range(5): 
        print(j)
        tr_te_sp = j
        index_train = np.load('index_tr.npy')[tr_te_sp].astype(int)-1
        index_test = np.load('index_te.npy')[tr_te_sp].astype(int)-1

        look_up_train_1 = look_up[index_train]
        grade_train_1 = grade[index_train]
        length_train_1 = length[index_train]
        weight_train_1 = weight[index_train]
        
        index_train, index_val = index_gen(0, 0.1, grade_train_1)
        look_up_train = look_up_train_1[index_train]
        print("training_data shape", look_up_train.shape)
        grade_train = grade_train_1[index_train]
        length_train = length_train_1[index_train]
        weight_train = weight_train_1[index_train]
        
        look_up_val = look_up_train_1[index_val]
        print("validation_data shape", look_up_val.shape)
        grade_val = grade_train_1[index_val]
        length_val = length_train_1[index_val]
        weight_val = weight_train_1[index_val]

        look_up_test = look_up[index_test]
        print("test_data shape", look_up_test.shape)
        grade_test = grade[index_test]
        length_test = length[index_test]
        weight_test = weight[index_test]
        
        
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay = 0.0)
        criterion = Loss_reference(0)
        '''
        train_qwk = KNN_pred_test_ws(model, look_up_train_1, length_train_1, grade_train_1, weight_train_1, num_class)
        print(train_qwk)
        '''
        
        val_best = 0

        for epoch in range(epoch_num):
            optimizer.zero_grad()
            anchor_out = model(look_up_train.cuda())

            loss_1 = criterion(anchor_out, length_train, grade_train, weight_train, model, num_class)
            loss = loss_1
            loss.backward()
            optimizer.step()

            #train_loss_acc[epoch][0] += loss.cpu().detach().numpy()
            print(epoch, "/", epoch_num)

            loss_ = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            
            if epoch > 10:
                val_acc = KNN_pred_test_ws(model, look_up_val, length_val, grade_val, weight_val, num_class)
                print("validation_accuracy", val_acc)
                if val_acc >= val_best:
                    val_best = val_acc
                    test_qwk = KNN_pred_test_ws(model, look_up_test, length_test, grade_test, weight_test, num_class)
                    #train_loss_acc[epoch][3] = test_qwk
                    print('test_accuracy', test_qwk, "model saved")
                    torch.save(model,'model'+str(tr_te_sp)+'_new.pkl')

            if epoch % 20 == 0:
                train_qwk = KNN_pred_test_ws(model, look_up_train, length_train, grade_train, weight_train, num_class)
                #train_loss_acc[epoch][1] = train_qwk
                print('train_accuracy', train_qwk)

            



if __name__ == '__main__':
    args = parser.parse_args()
    look_up, grade, length, weight = DataLoader()
    weight = weight.cuda()
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    model = conv_ls_w(300, 300, int(args.number), 5)
    model.cuda()
        
    num_class = 5
    #model = torch.load('model'+str(seed)+'_new.pkl')
    
    train(model, args, look_up, grade, length, weight, num_class)




