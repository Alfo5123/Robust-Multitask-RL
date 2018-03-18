import torch
import torch.nn as nn
from a3cutils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
import numpy as np


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 50)
        self.pi2 = nn.Linear(50,50)
        self.pi3 = nn.Linear(50, a_dim)
        self.v1 = nn.Linear(s_dim, 50)
        self.v2 = nn.Linear(50,50)
        self.v3 = nn.Linear(50, 1)
        set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        logits = self.pi3(F.relu(self.pi2(pi1)))
        v1 = F.relu(self.v1(x))
        values = self.v3(F.relu(self.v2(v1)))
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):

    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, \
                 update_global_iter , num_episodes , max_num_steps_per_episode, \
                 gamma, env, ns, na ):

        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt

        self.ns = ns
        self.na = na

        self.lnet = Net(ns, na)           # local network
        self.env = env

        self.update_global_iter = update_global_iter
        self.num_episodes = num_episodes
        self.max_num_steps_per_episode = max_num_steps_per_episode
        self.gamma = gamma


    def run(self):

        total_step = 1
        while self.g_ep.value < self.num_episodes:
            #s = np.reshape( self.env.reset() , ( self.ns, 1 ) ).flatten()  ## Line to fix for arbitrary
            s = self.env.reset()

            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.0

            for _ in range(self.max_num_steps_per_episode):

                a = self.lnet.choose_action(v_wrap(s[None,:]))
                s_, r, done, _ = self.env.step(a)
                #s_ = np.reshape( s_ , ( self.ns, 1 ) ).flatten()  ## Line to fix for arbitrary environment

                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.update_global_iter == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, self.gamma)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


def trainA3C(file_name="A3C", env=GridworldEnv(1), update_global_iter=10,
            gamma=0.999, is_plot=False, num_episodes=500, 
            max_num_steps_per_episode=1000, learning_rate=0.0001 ):

    """
    A3C training routine. Retuns rewards and durations logs.
    Plot environment screen
    """
    ns = env.observation_space.shape[0]  ## Line to fix for arbitrary environment
    na = env.action_space.n

    gnet = Net(ns, na)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr = learning_rate )      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, update_global_iter, num_episodes , max_num_steps_per_episode, gamma, env, ns, na ) for i in range(mp.cpu_count())]

    [w.start() for w in workers]
    episode_rewards = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            episode_rewards.append(r)
        else:
            break
    [w.join() for w in workers]

    #Store results
    np.save(file_name + '-a3c-rewards', episode_rewards)


    return episode_rewards
