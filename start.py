# training
# author:fanzhe


import torch
import os,random
import numpy as np
import torch.nn as nn
from collections import deque
from DQN.Brain import Brain
from Handle import Handle

# Hyper Parameters


LR = 0.001                  # learning rate
Epsilon = 0.9               # greedy policy
Gamma = 0.9                 # reward discount
Batch_size = 32
Target_replace_iter = 32    # target net update frequency
Memory_capacity = 64        # total memory
States = 4                  # state Action Reward state_next
Actions = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepQN(object):
    def __init__(self):
        self.device = device
        self.eval_net = Brain().to(self.device)
        self.target_net = Brain().to(self.device)
        self.replay_memory = deque()
        self.memory_capacity = Memory_capacity
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.batch_size = Batch_size
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self,views):
        views = torch.unsqueeze(torch.FloatTensor(views),0).to(device)
        if np.random.uniform()<Epsilon: # greedy
            actions_value = self.eval_net.forward(views)
            action = torch.max(actions_value,1)[1].cpu().data.numpy()  # 改动点
            action = action[0]
        else:
            action = np.random.randint(0,Actions)

        return action

    def store_transition(self, state, Action, Reward, state_next):

        next_state = state_next
        self.replay_memory.append((state, Action, Reward, next_state, terminal))
        if len(self.replay_memory) > self.memory_capacity:
            self.replay_memory.popleft()

        self.memory_counter +=1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter +=1

        # sample batch transitions

        minibatch = random.sample(self.replay_memory, self.batch_size)
        b_state = torch.FloatTensor(np.array([data[0] for data in minibatch])).to(device)
        b_action = torch.LongTensor(np.array([[data[1]] for data in minibatch])).to(device)
        b_reword = torch.FloatTensor(np.array([data[2] for data in minibatch])).to(device)
        b_state_next = torch.FloatTensor(np.array([data[3] for data in minibatch])).to(device)


        q_eval = self.eval_net(b_state).gather(1,b_action)
        q_next = self.target_net(b_state_next).detach()
        q_target = b_reword + Gamma * q_next.max(1)[0].view(self.batch_size,1)

        loss = self.loss_func(q_eval,q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.target_net, './model/model_gpu.check')

    def load(self):
        self.eval_net = torch.load('./model/model_gpu.check')
        self.target_net = torch.load('./model/model_gpu.check')
        print('load model succes...')


dqn = DeepQN()
Op = Handle()

print('\n collecting experience...')

if os.path.exists('./model/model_gpu.check'):
    dqn.load()

total_step = 0
for i_episode in range(1000):
    Op.getstate()
    while True:

        action = dqn.choose_action(Op.state)
        # 执行行为
        state_next,reward,terminal = Op.action(action)

        if terminal:
            break

        dqn.store_transition(Op.state,action,reward,state_next)


        if dqn.memory_counter>Memory_capacity:
            dqn.learn()
            print(f'Ep:{i_episode} | Ep_r:{round(reward,3)} | total_step:{total_step}')

        if i_episode % 50 == 0:
            dqn.save()

        total_step+=1

        Op.state = state_next













