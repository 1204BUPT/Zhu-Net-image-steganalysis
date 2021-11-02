import torch
import torch.nn as nn
import visdom
import os
import torch.optim as optim
import numpy as np
from functools import reduce
import operator
from statistics import statistics


class Trainer(object):
    '''
    Args:
        model: network
        lr: learning rate
        lr_decay: leaning_rate decay
        token: visdom_name
        token1: visdom_win_name
        optimizer: Default(None)
        save_dir: Default(None)
        save_freq: Default(5)
        cur_epoch: 0当前批次
        print_freq: 150

    '''


    def __init__(self, model, lr, lr_decay, token,token1, weight_decay,save_dir,optimizer=None,
                 save_freq=1, cur_epoch=0,print_freq=150, shedule_lr = None):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.count = 1
        self.count2 = 1
        self.loss_f = nn.CrossEntropyLoss().to(self.device)
        self.shedule_lr = shedule_lr

        self.lr_decay = lr_decay
        if optimizer==None:

            self.optimizer = optim.SGD(self.model.layer2.parameters(), lr=lr, momentum=0.95, weight_decay=weight_decay)

        else:
            self.optimizer = optimizer


  

        self.save_dir = "PATH" + save_dir

        isExists = os.path.exists(self.save_dir)

        if not isExists:
            os.makedirs(self.save_dir)



        self.save_freq = save_freq
        self.print_freq = print_freq
        self.cur_epoch = cur_epoch
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

        self.best_acc = 0

    def train(self, train_loader):
        """
        Args:
 ,
        :param train_loader:

        :return: None
        """

        self.model.train()

        running_loss = 0.
        self.cur_epoch += 1

        print("Epoch:", self.cur_epoch)
        if self.cur_epoch in self.shedule_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= 5
            print (param_group['lr'])

        for batch_idx, (data, labels) in enumerate(train_loader):

            data, labels = data.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)

            loss = self.loss_f(outputs, torch.max(labels, 1)[0])

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.cur_epoch, (batch_idx + 1) * len(data), len(train_loader.dataset) * 2,
                                    100. * (batch_idx + 1) / (len(train_loader)), loss.item()))

                self.count += 1

        running_loss /= len(train_loader)
        self.train_loss.append(running_loss)

        self.count2 += 1


        if ((self.cur_epoch) % self.save_freq == 0):
            torch.save(self.model.state_dict(),
                       self.save_dir + "/" + "epoch4.13：150+_" + str(
                           self.cur_epoch) + ".pkl")

    def valid(self, valid_loader):

        

        self.model.eval()

        valid_loss = 0
        correct = 0
        for data, labels in valid_loader:

            data, labels = data.to(self.device), labels.to(self.device)

            output = self.model(data)

            valid_loss_temp = self.loss_f(output, torch.max(labels, 1)[0])  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

            valid_loss += valid_loss_temp.item()
            
            correct += pred.eq(labels.data.view_as(pred)).sum()
            

        valid_loss /= len(valid_loader)
        
        cur_acc = 100. * (correct.item()) / (len(valid_loader.dataset) * 2)
        print("Valid Loss: {}".format(valid_loss))
        self.val_acc.append(cur_acc)
        self.val_loss.append(valid_loss)
        if (cur_acc > self.best_acc):
            self.best_acc = cur_acc
            torch.save(self.model.state_dict(),
                       self.save_dir + "/" + "epoch_" + str(
                           self.cur_epoch) + "_best_acc_" + str(cur_acc) + ".pkl")

        self.vis.line(X=np.array([self.cur_epoch]), Y=np.array([valid_loss]), win=self.token1 + "1" + "valid",
                      update=None if self.cur_epoch == 1 else 'append',
                      opts={'xlabel': 'epoch_'+self.token1, 'ylabel': 'Valid loss'})
        self.vis.line(X=np.array([self.cur_epoch]), Y=np.array([cur_acc]), win=self.token1 + "2" + "valid",
                      update=None if self.cur_epoch == 1 else 'append',
                      opts={'xlabel': 'epoch_'+self.token1, 'ylabel': 'Valid Acc'})

    def save_loss_val_acc(self):
        fp = open(self.save_dir + "/" + "loss.txt", 'a')
        for i in self.train_loss:
            fp.write(str(i) + ',')
        fp.close()

        fp = open(self.save_dir + "/" + "val_acc.txt", 'a')
        for i in self.val_acc:
            fp.write(str(i) + ',')
        fp.close()
        
        fp = open(self.save_dir + "/" + "val_loss.txt", 'a')
        for i in self.val_loss:
            fp.write(str(i) + ',')
        fp.close()

    def test(self, test_loader):
        self.model.eval()

        test_loss = 0
        correct = 0
        for data, labels in test_loader:

            data, labels = data.to(self.device), labels.to(self.device)

            output,FEATURE= self.model(data)#问题在此，少一个feature

            # test_loss += self.loss_f(output, torch.max(labels, 1)[0]).item()  # sum up batch loss
            test_loss += self.loss_f(output, torch.max(labels, 1)[0]).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        print('\nTest set: Total loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                   len(test_loader.dataset) * 2,
                                                                                   100. * correct / (len(
                                                                                       test_loader.dataset) * 2),
                                                                                   test_loss))
