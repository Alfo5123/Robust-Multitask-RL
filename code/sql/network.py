import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory_replay import Transition

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DQN(nn.Module):
    """
    Deep neural network with represents an agent.
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(10)
        self.head = nn.Linear(200, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

# class DQN(nn.Module):
#     """
#     Deep neural network with represents an agent.
#     """
#     def __init__(self, num_actions):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=2)
#         self.max_pool = nn.MaxPool2d((2,2))
#         self.bn1 = nn.BatchNorm2d(10)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(20)
#         self.linear = nn.Linear(80, 20)
#         # self.bn3 = nn.BatchNorm1d(50)
#         self.head = nn.Linear(20, num_actions)

#     def forward(self, x):
#         x = F.leaky_relu(self.max_pool(self.bn1(self.conv1(x))))
#         x = F.leaky_relu(self.bn2(self.conv2(x)))
#         x = F.leaky_relu(self.linear(x.view(x.size(0), -1)))
#         return self.head(x)

def select_action(state, model, num_actions,
                    EPS_START, EPS_END, EPS_DECAY, steps_done):
    """
    Selects whether the next action is choosen by our model or randomly
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(num_actions)]])


def optimize_model(model, optimizer, memory, BATCH_SIZE, GAMMA, BETA):
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = torch.log( torch.exp(
                        BETA * model(non_final_next_states)).sum(1)) / BETA
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
