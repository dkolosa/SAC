import os
import torch as T
import numpy as np
from modeel import Actor, Critic, Value
from replay_memory import Uniform_Memory

class SAC:
    def __init__(self, states, actions, action_bound, batch_size, nn_dims, lr, gammaa, tau, save_dir):

        self.states = states
        self.action = actions
        self.action_bound = action_bound
        self.batch_size = batch_size

        layer_1, layer_2 = nn_dims
        lr_act, lr_crit, lr_val = lr

        self.gammaa = gamma
        self.tau = tau
        self.save_dir = save_dir

        self.actor = Actor(self.states, self.action, self.action_bound, layer_1, layer_2, lr_act)
        self.critic = Critic(self.states, self.action, layer_1, layer_2, lr_crit)
        self.value = Value(self.states, layer_1, layer_2, lr_val)

        self.critic_target = Critic(self.states, self.action, layer_1, layer_2, lr_crit)
        self.value_target = Value(self.states, layer_1, layer_2, lr_val)

        self.replay = Uniform_Memory(buffer_size=100000)

        self.update_target(self.tau)

    
    def action(self, a):
        pass

    def train(self):
        pass

    def update_target(self, tau=.001):
        pass

