import torch
import torch.nn.functional as F
from load_data import My_data
import os
import numpy as np 

FILE = 'srcnn.t7'
np.set_printoptions(precision=7)

class SRCNN(torch.nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()
        self.Pear = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 9))
        self.Nlm = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1))
        self.Rec = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(5, 5))

    def forward(self, x):
        x = F.relu(self.Pear(x))
        x = F.relu(self.Nlm(x))
        x = self.Rec(x)
        return x

    def train(self, iteration, loader):
        loss_list = []
        criter = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lr=1e-6, params=self.parameters())
        for _ in range(iteration):
            print('iteration:', _, end='   ')
            sum_loss = 0
            for step, (x, y) in enumerate(loader):
                predict = self(x.cuda())
                loss = criter(predict, y.cuda())
                optimizer.zero_grad()
                sum_loss += loss
                loss.backward()
                optimizer.step()
            print('loss:', sum_loss / step)
            loss_list.append(sum_loss / step)
            if _ % 10 == 0:
                torch.save(self, 'srcnn.t7')
        np.save('loss.npy', loss_list)

