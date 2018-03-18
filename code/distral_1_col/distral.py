import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch distral example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Distral(nn.Module):

    def __init__(self, tasks = 2 ):

        super(Distral, self).__init__()
        self.affines = torch.nn.ModuleList ( [ nn.Linear(4, 128) for i in range(tasks+1) ] )
        self.action_head = torch.nn.ModuleList ( [ nn.Linear(128, 2) for i in range(tasks+1) ] )
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = [[] for i in range(tasks+1)] 
        self.rewards = [[] for i in range(tasks+1)] 
        self.tasks = tasks 

    def forward(self, x):

        action_scores = torch.cat( [ F.softmax(self.action_head[i](F.relu(self.affines[i](x))), dim=-1) for i in range(self.tasks+1) ])
        state_values = self.value_head(F.relu(self.affines[0](x)))
        return action_scores.view(self.tasks+1,-1) , state_values


model = Distral( )
optimizer = optim.Adam(model.parameters(), lr=3e-2)


def select_action(state, tasks):
    state = torch.from_numpy(state).float()
    probs, state_value = model(Variable(state))

    # Obtain the most probable action for each one of the policies
    actions = []
    for i in range(tasks+1):
        m = Categorical(probs[i])
        actions.append( m.sample() )
        model.saved_actions[i].append(SavedAction(m.log_prob(actions[i]), state_value))

    return torch.cat( actions )


def finish_episode( tasks , alpha , beta , gamma ):

    ### Calculate loss function according to Equation 1
    R = 0
    saved_actions = model.saved_actions[1]
    policy_losses = []
    value_losses = []

    ## Obtain the discounted rewards backwards
    rewards = []
    for r in model.rewards[1][::-1]:
        R = r + gamma * R 
        rewards.insert(0, R)

    ## Standardize the rewards to be unit normal (to control the gradient estimator variance)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))


    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()

    #Clean memory
    for i in range(tasks+1):
        del model.rewards[i][:]
        del model.saved_actions[i][:]


def trainDistral( file_name="Distral", envs=[gym.make('CartPole-v0').unwrapped], 
            batch_size=128, alpha = 0.5 , beta = 0.5, gamma=0.999, is_plot=False,
            num_episodes=500, max_num_steps_per_episode=10000, learning_rate=0.001 ):

    #Run each one of the policies for the different environments
    #update the policies

    tasks = len(envs)

    for env in envs:  ### From task 1 to N
        for i_episode in count(1):
            total_reward = 0
            state = env.reset()
            for t in range(max_num_steps_per_episode):  # Don't infinite loop while learning
                action = select_action(state, tasks )
                state, reward, done, _ = env.step(action.data[1])
                if is_plot:
                    env.render()
                model.rewards[1].append(reward)
                total_reward += reward
                if done:
                    break

            finish_episode( tasks , alpha , beta, gamma )
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}\tTotal Reward: {:.2f}'.format(
                    i_episode, t, total_reward))
            if total_reward > 200:
                print("Solved! Total reward is now {} and "
                      "the last episode runs to {} time steps!".format(total_reward, t))
                break


if __name__ == '__main__':
    trainDistral()