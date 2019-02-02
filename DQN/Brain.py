# DQN 神经网络
# date: 2019.1.31 author:fanzhe

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import deque

Actions = 2 # 游戏的两种行为

class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()

        self.replay_memory = deque()
        self.actions = Actions
        self.mem_size = 300


        self.conv1 = nn.Conv2d(4,32,kernel_size=4,stride=2,padding=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3)

        self.fc1 = nn.Linear(64*57*57,256)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc2  = nn.Linear(256,self.actions)


    def forward(self, input):
        out = self.net(input)
        out = out.view(out.size()[0],-1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        return out



if __name__ == '__main__':
    import numpy as np

    network = Brain()
    single_pic = np.zeros((460,460),dtype=np.float32)
    view = np.stack((single_pic,single_pic,single_pic,single_pic),axis=0)
    view = Variable(torch.from_numpy(view)).unsqueeze(0)  # 注意

    output = network(view)

    actions = torch.max(output, 1)[1].data.numpy()

    print(f'output: {output}')
    print(f'action :{actions}')

    print(output.size())

    # print(network)