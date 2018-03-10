import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch
import math
import numpy as np
from memory_replay import ReplayMemory, Transition
from network import DQN, select_action, optimize_model, Tensor
import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv


def plot_durations(episode_durations, mean_durations):
    plt.figure(2)
    plt.clf()
    # durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations, label="durations")
    # Take 100 episode averages and plot them too
    # if len(episode_durations) >= 100:
        # means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        # means = torch.cat((torch.zeros(99), means))
    mean_durations.append(np.mean(np.array(episode_durations)[::-1][:100]))
    plt.plot(mean_durations, label="means")
    plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())

def plot_rewards(episode_rewards, mean_rewards):
    plt.figure(3)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards, label="rewards")
    mean_rewards.append(np.mean(np.array(episode_rewards)[::-1][:100]))
    plt.plot(mean_rewards, label="means")
    plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())

def plot_state(state):
    if state is not None:
        plt.figure(1)
        plt.clf()
        plt.imshow(state.cpu().squeeze(0).squeeze(0).numpy(),
                       interpolation='none')
        plt.draw()
        plt.pause(0.00001)

def trainSQL(file_name="SQL", env=GridworldEnv(1), batch_size=128,
            gamma=0.999, beta=5, eps_start=0.9, eps_end=0.05, eps_decay=1000,
            is_plot=False, num_episodes=500, max_num_steps_per_episode=1000,
            learning_rate=0.001, memory_replay_size=10000):
    """
    Soft Q-learning training routine. Retuns rewards and durations logs.
    Plot environment screen
    """

    def get_screen():
    # TODO: may have some bugs
        screen = env.current_grid_map
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = torch.from_numpy(screen)
        return screen.unsqueeze(0).unsqueeze(0).type(Tensor)
    if is_plot:
        env.reset()
        plt.ion()
        plt.figure()
        plt.imshow(get_screen().cpu().squeeze(0).squeeze(0).numpy(),
                   interpolation='none')
        plt.draw()
        plt.pause(0.00001)

    num_actions = env.action_space.n
    model = DQN(num_actions)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    memory = ReplayMemory(memory_replay_size)

    episode_durations = []
    mean_durations = []
    episode_rewards = []
    mean_rewards = []

    steps_done = 0
    for i_episode in range(num_episodes):
        print("Cur episode:", i_episode, "steps done:", steps_done,
                "exploration factor:", eps_end + (eps_start - eps_end) * \
                math.exp(-1. * steps_done / eps_decay))
        # Initialize the environment and state
        env.reset()
        # last_screen = env.current_grid_map
        current_screen = get_screen()
        state = current_screen # - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state, model, num_actions,
                                    eps_start, eps_end, eps_decay, steps_done)
            steps_done += 1
            _, reward, done, _ = env.step(action[0, 0])
            reward = Tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen # - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            # plot_state(state)
            # env.render()

            # Perform one step of the optimization (on the target network)
            optimize_model(model, optimizer, memory, batch_size, gamma, beta)
            if done or t + 1 >= max_num_steps_per_episode:
                episode_durations.append(t + 1)
                episode_rewards.append(env.episode_total_reward)
                if is_plot:
                    plot_durations(episode_durations, mean_durations)
                    plot_rewards(episode_rewards, mean_rewards)
                break

    print('Complete')
    env.render(close=True)
    env.close()
    if is_plot:
        plt.ioff()
        plt.show()

    ## Store Results

    np.save(file_name + '-sql-rewards', episode_rewards)
    np.save(file_name + '-sql-durations', episode_durations)

    return episode_rewards, episode_durations
