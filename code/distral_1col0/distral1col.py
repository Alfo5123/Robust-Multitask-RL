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
parser.add_argument('--seed', type=int, default=512, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='interval between training status logs (default: 5)')
args = parser.parse_args()

import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv

#env = gym.make('CartPole-v0')
#env.seed(args.seed)


#torch.manual_seed(args.seed)


class Policy(nn.Module):

    def __init__(self, input_size, num_actions ):

        super(Policy, self).__init__()
        self.affines = nn.Linear(input_size, 100)
        self.action_head = nn.Linear(100, num_actions) 

        self.saved_actions = [] 
        self.rewards = [] 

    def forward(self, x):
        action_scores = F.softmax(self.action_head(F.relu(self.affines(x))), dim=-1) 
        return action_scores



def select_action(state, policy , distilled ):

    # Format the state
    state = torch.from_numpy(state).float()

    # Run the policy
    probs = policy(Variable(state))

    # Obtain the most probable action for the policy
    m = Categorical(probs)
    action =  m.sample() 
    policy.saved_actions.append( m.log_prob(action))


    # Run distilled policy
    probs0 = distilled(Variable(state))

    # Obtain the most probably action for the distilled policy
    m = Categorical(probs0)
    action_tmp =  m.sample() 
    distilled.saved_actions.append( m.log_prob(action_tmp) )

    # Return the most probable action for the policy
    return action


def finish_episode( policy, distilled, opt_policy , opt_distilled, alpha, beta , gamma):

    ### Calculate loss function according to Equation 1

    ## Store three type of losses
    reward_losses = []
    distill_losses = []
    entropy_losses = []

    # Give format
    alpha = Variable(torch.Tensor([alpha]))
    beta = Variable(torch.Tensor([beta]))

    # Retrive distilled policy actions
    distill_actions = distilled.saved_actions

    # Retrieve policy actions and rewards
    policy_actions = policy.saved_actions
    rewards = policy.rewards

    # Obtain discounts 
    R = 1.
    discounts = []
    for r in policy.rewards[::-1]:
        R *= gamma 
        discounts.insert(0,R)

    discounts = torch.Tensor(discounts)
    #print(discounts)

    for log_prob_i, log_prob_0, d , r in zip(policy_actions, distill_actions, discounts, rewards ):
        reward_losses.append( -d * Variable(torch.Tensor([r]) ) )
        distill_losses.append( -( (d*alpha)/beta ) * log_prob_0 )
        entropy_losses.append( (d/beta) * log_prob_i )



    #print('Reward Loss: ',torch.stack(reward_losses).sum().data[0])
    #print('Entropy Loss: ',torch.stack(entropy_losses).sum().data[0])
    #print('Distill Loss: ',torch.stack(distill_losses).sum().data[0])

    # Perform optimization step
    opt_policy.zero_grad()
    opt_distilled.zero_grad()

    loss = torch.stack(reward_losses).sum() + torch.stack(entropy_losses).sum() + torch.stack(distill_losses).sum()
    
    loss.backward(retain_graph=True)


    #for param in policy.parameters():
    #    param.grad.data.clamp_(-1, 1)

    opt_policy.step()
    opt_distilled.step()

    #Clean memory
    del policy.rewards[:]
    del policy.saved_actions[:]
    del policy.saved_actions[:]


def trainDistral( file_name="Distral_1col", list_of_envs=[GridworldEnv(5), GridworldEnv(4)], batch_size=128, gamma=0.80, alpha=0.5,
            beta=0.005, is_plot=False, num_episodes=1000,
            max_num_steps_per_episode=10, learning_rate=0.001,
            memory_replay_size=10000, memory_policy_size=1000 ):
    

    # Specify Environment conditions
    input_size = list_of_envs[0].observation_space.shape[0]
    num_actions = list_of_envs[0].action_space.n
    tasks = len(list_of_envs)

    # Define our set of policies, including distilled one
    models = torch.nn.ModuleList( [Policy(input_size, num_actions) for _ in range(tasks+1)] )
    optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

    # Store the total rewards
    episode_rewards = [ [] for i in range(num_episodes) ]
    episode_duration = [ [] for i in range(num_episodes) ]

    for i_episode in range(num_episodes):

        # For each one of the envs
        for i_env, env in enumerate(list_of_envs):

            #Initialize state of envs
            state = env.reset()

            #Store total reward per environment per episode
            total_reward = 0

            # Store duration of each episode per env
            duration = 0
            
            for t in range(max_num_steps_per_episode):

                # Run our policy
                action = select_action(state, models[i_env + 1], models[0] )

                next_state, reward, done, _ = env.step(action.data[0])
                models[i_env+1].rewards.append(reward)
                total_reward += reward
                duration += 1

                #if is_plot:
                #    env.render()

                if done:
                    break

                #Update state
                state = next_state

            episode_rewards[i_episode].append(total_reward) 
            episode_duration[i_episode].append(duration) 

            # Distill for each environment
            finish_episode(models[i_env + 1], models[0], optimizers[i_env + 1], optimizers[0] , alpha , beta, gamma )


        if i_episode % args.log_interval == 0:
            for i in range(tasks):
                print('Episode: {}\tEnv: {}\tDuration: {}\tTotal Reward: {:.2f}'.format(
                    i_episode, i, episode_duration[i_episode][i], episode_rewards[i_episode][i]))

    np.save(file_name + '-distral0-rewards' , episode_rewards)
    np.save(file_name + '-distral0-duration' , episode_duration)

    print('Completed')

if __name__ == '__main__':
	trainDistral()