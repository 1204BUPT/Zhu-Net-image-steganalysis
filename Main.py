from utils_for_dataset import getDataLoader
from utils_train_valid import Trainer

from Zhu-Net import Zhu_Net

import time
import torch
import time
import torch

train_loader, valid_loader, test_loader = getDataLoader(
    r'PATH',# training set, cover
    r'PATH',# training set, stego
    r'PATH',# testing set, cover
    r'PATH',# testing set, stego
    r'PATH',# valid set, cover
    r'PATH',# valid set, stego
    1
)
net = Zhu_Net()

trainer = Trainer(model=net, lr=0.001, cur_epoch=0, lr_decay=0.95, weight_decay=0.0, shedule_lr=[20, 35, 50, 65],
                  token='Best_biggest_hill_0.1', token1='hill_0.1_trueD', save_dir="Best_biggest_hill_0.1_trueD",
                  print_freq=150)

for cur_epoch in range(150):
    time1 = time.time()
    trainer.train(train_loader=train_loader)
    time2 = time.time()
    print("epoch time: {}".format(time2 - time1))
torch.save(net, r'PATH')
