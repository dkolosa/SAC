import datetime
import os
import random

import gym
import numpy as np

from sac import SAC


if __name__ == '__main__':

    num_episodes = 1000
    iter_per_episode = 200  # seconds used for early stopping

    ENV = 'Pendulum-v0'
    env = gym.make(ENV)

    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
    save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

    n_action = env.action_space.shape[0]
    n_states = env.observation_space.shape[0]
    action_bound = env.action_space.high
    batch_size = 10
    # Will have to add conv nets for processing
    # use conv and FC layers to process the images

    # use FC layers to process the current speed
    layer_nodes = [128, 128]

    tau = 0.01
    lr = [0.0001, 0.001, 0.001]
    GAMMA = 0.99

    agent = SAC(n_states, n_action, action_bound,batch_size, layer_nodes, lr, GAMMA,tau, save_dir)

    load_models = False
    save = False

    if load_models:
        agent.load_model()

    j = 0
    for i in range(num_episodes):
        s = env.reset()
        done = False
        sum_reward = 0
        while True:
            # env.render()
            a = agent.take_action(s)
            s1, r, done, _ = env.step(a)

            # Store in replay memory
            agent.memory.add(
                (np.reshape(s, (n_states,)), np.reshape(a, (n_action,)), r, 
                np.reshape(s1, (n_states,)), done))

            agent.train()

            sum_reward += r
            s = s1
            j += 1
            if done:
                if save:
                    agent.save_model()
                print(f'Episode {i} of {num_episodes}, reward {sum_reward:.4f} '
                        f'\n==================================================================')
                break
