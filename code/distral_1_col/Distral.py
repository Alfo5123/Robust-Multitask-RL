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
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='interval between training status logs (default: 5)')
args = parser.parse_args()

import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv

#env = gym.make('CartPole-v0')
#env.seed(args.seed)
torch.manual_seed(args.seed)


class Distral(nn.Module):

    def __init__(self, tasks = 2 ):

        super(Distral, self).__init__()
        self.affines = torch.nn.ModuleList ( [ nn.Linear(3, 100) for i in range(tasks+1) ] )
        self.action_head = torch.nn.ModuleList ( [ nn.Linear(100, 5) for i in range(tasks+1) ] )

        self.saved_actions = [[] for i in range(tasks+1)] 
        self.rewards = [[] for i in range(tasks)] 
        self.tasks = tasks 

    def forward(self, x):
        x = x.view( self.tasks + 1, -1 )
        action_scores = torch.cat( [ F.softmax(self.action_head[i](F.relu(self.affines[i](x[i]))), dim=-1) for i in range(self.tasks+1) ])
        return action_scores.view(self.tasks+1,-1) 


model = Distral( )
optimizer = optim.Adam(model.parameters(), lr=3e-3)


def select_action(state, tasks):
    state = torch.from_numpy(state).float()
    probs = model(Variable(state))

    # Obtain the most probable action for each one of the policies
    actions = []
    for i in range(tasks+1):
        m = Categorical(probs[i])
        actions.append( m.sample() )
        model.saved_actions[i].append( m.log_prob(actions[i]))

    return torch.cat( actions )


def finish_episode( tasks , alpha , beta , gamma ):

    ### Calculate loss function according to Equation 1

    ## Store three type of losses
    reward_losses = []
    distill_losses = []
    entropy_losses = []

    # Give format
    alpha = Variable(torch.Tensor([alpha]))
    beta = Variable(torch.Tensor([beta]))

    # Retrive distilled policy actions
    distill_actions = model.saved_actions[0]
    #print(distill_actions)

    # Calculate the sum of losses for each policy
    for i in range(tasks):

        # Retrieve lopprob actions
        saved_actions = model.saved_actions[i+1]

        ## Obtain discounts backwards
        R = 1.
        discounts = []
        for r in model.rewards[i][::-1]:
            R *= gamma 
            discounts.append(R)

        discounts = torch.Tensor(discounts)

        for log_prob_i, log_prob_0, d , r in zip(saved_actions, distill_actions, discounts, model.rewards[i]):
            reward_losses.append( -d * Variable(torch.Tensor([r])  ) )
            distill_losses.append( -( (d*alpha)/beta ) * log_prob_0 )
            entropy_losses.append( (d/beta) * log_prob_i )

    #print('Reward Loss: ',torch.stack(reward_losses).sum().data[0])
    #print('Entropy Loss: ',torch.stack(entropy_losses).sum().data[0])
    #print('Distill Loss: ',torch.stack(distill_losses).sum().data[0])

    # Perform optimization step
    optimizer.zero_grad()
    loss = torch.stack(reward_losses).sum() + torch.stack(entropy_losses).sum() + torch.stack(distill_losses).sum()
    
    loss.backward()
    #for param in model.parameters():
    #    param.grad.data.clamp_(-1, 1)

    optimizer.step()

    #Clean memory
    for i in range(tasks):
        del model.rewards[i][:]
    for i in range(tasks+1):
        del model.saved_actions[i][:]



def trainDistral( file_name="Distral", envs=[GridworldEnv(4),GridworldEnv(5)], 
            alpha = 0.0, beta = 0.5, gamma=0.95, is_plot=False,
            num_episodes=100, max_num_steps_per_episode=50, learning_rate=0.001 ):

    #Run each one of the policies for the different environments
    #update the policies

    tasks = len(envs)

    for i_episode in range(num_episodes):

        total_reward = 0

        #Initialize state of envs
        global_state = []
        for i in range(tasks): 
            state = envs[i].reset()
            global_state.append( state )


        for t in range(max_num_steps_per_episode):  # Don't infinite loop while learning

            print(torch.from_numpy(np.asarray(global_state)))
            actions = select_action(np.asarray(global_state), tasks )


            state, reward, done, _ = env.step(actions.data[i+1])


            #if is_plot:
            #    env.render()


            model.rewards[i].append(reward)
            total_reward += reward
            if done:
                break
            #print(total_reward)

        finish_episode( tasks , alpha , beta, gamma )
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tTotal Reward: {:.2f}'.format(
                    i_episode, t, total_reward))
        if total_reward > 0.9:
            print("Solved! Total reward is now {} and "
                    "the last episode runs to {} time steps!".format(total_reward, t))
            break


if __name__ == '__main__':
    trainDistral()