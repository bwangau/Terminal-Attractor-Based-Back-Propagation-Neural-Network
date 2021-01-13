#A regression test on TABP neural network, written in Pytorch, by Bin Wang
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd

sns.set_style(style = 'whitegrid')
plt.rcParams["patch.force_edgecolor"] = True

#create a single training set
m = 2
c = 3
x = np.random.rand(256)
noise = np.random.randn(256)/4
y=m*x+c+noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y
sns.lmplot(x = 'x', y = 'y', data = df)

import torch
import torch.nn as nn
from torch.autograd import Variable
x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
#learning rate
lr = 0.01 
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
[w,b] = model.parameters()

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer

def get_param_values():
    return w.data[0][0], b.data[0]

def plot_current_fit(title = ""):
    plt.figure(figsize = (12, 4))
    plt.title(title)
    plt.scatter(x, y, s = 8)
    w1 = w.data[0][0]
    b1 = b.data[0]
    x1 = np.array([0., 1.])
    y1 = w1*x1 + b1
    plt.plot(x1, y1, 'r', label = "current fit ({:.3f}, {:.3f})".format(w1, b1))
    plt.xlabel('x (input)')
    plt.ylabel('y (target)')
    plt.legend()
    plt.show()

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
#the parameter rho in TABP
rho = 1/3 
for epoch in range(500): 
    y_pred = model(x_train)
    train_loss = criterion(y_pred, y_train)
    print('lr: ', param_group['lr'])
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    o=[]
    for param_group in optimizer.param_groups:
        for param_p in param_group['params']:
            o.append(param_p.grad.view(-1))
        re = torch.cat(o)
        #implement the training algorithm
        param_group['lr'] = lr*train_loss.item()**rho/torch.norm(re)
    print('epoch: ', epoch, ' loss: ', train_loss.item())

[w,b] = model.parameters()
plot_current_fit('After training')
