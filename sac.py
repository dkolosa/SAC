import os
import torch as T
import torch.nn.functional as F
import numpy as np
from modeel import Actor, Critic, Value
from replay_memory import Uniform_Memory

class SAC:
    def __init__(self, states, actions, action_bound, batch_size, nn_dims, lr, gamma, tau, save_dir):

        self.states = states
        self.action = actions
        self.action_bound = action_bound
        self.batch_size = batch_size

        layer_1, layer_2 = nn_dims
        lr_act, lr_crit, lr_val = lr

        self.reward_scale = 2
        self.gamma = gamma
        self.tau = tau
        self.save_dir = save_dir
    
        self.actor = Actor(self.states, self.action, self.action_bound, layer_1, layer_2, lr_act)
        self.critic = Critic(self.states, self.action, layer_1, layer_2, lr_crit)
        self.value = Value(self.states, layer_1, layer_2, lr_val)

        self.critic_target = Critic(self.states, self.action, layer_1, layer_2, lr_crit)
        self.value_target = Value(self.states, layer_1, layer_2, lr_val)

        self.memory = Uniform_Memory(buffer_size=100000)

        self.update_target(self.tau)

    
    def take_action(self, state):
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        act, _ = self.actor.normaalize_sample(state, reparam=False)
        return act.cpu().detach().numpy()[0]

    def train(self):
        if self.batch_size < self.memory.get_count:
            mem = self.memory.sample(self.batch_size)
            s_rep = T.tensor(np.array([_[0] for _ in mem]), dtype=T.float).to(self.actor.device)
            a_rep = T.tensor(np.array([_[1] for _ in mem]), dtype=T.float).to(self.actor.device)
            r_rep = T.tensor(np.array([_[2] for _ in mem]), dtype=T.float).to(self.actor.device)
            s1_rep = T.tensor(np.array([_[3] for _ in mem]), dtype=T.float).to(self.actor.device)
            d_rep = T.tensor(np.array([_[4] for _ in mem])).to(self.actor.device)


            # Calculate critic and train
            value = self.value.forwaard(s_rep).view(-1)
            value_1 = self.value_target.forwaard(s1_rep).view(-1)
            value_1[d_rep] = 0.0

            # Value loss
            # probs, q = self.calculate_q(s_rep,reparam=False)
            acts, probs = self.actor.normaalize_sample(s_rep,reparam=False)
            probs = probs.view(-1)
            q_1 = self.critic.forwaard(s_rep, acts)
            q_2 = self.critic_target.forwaard(s_rep, acts)
            q = T.min(q_1, q_2)
            q = q.view(-1)

            self.value.optimizer.zero_grad()
            value_target = q - probs
            value_loss = .5 * F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()

            # actor loss
            # probs_act, q_act = self.calculate_q(s_rep, reparam=True)
            acts, probs = self.actor.normaalize_sample(s_rep,reparam=True)
            probs = probs.view(-1)
            q_1 = self.critic.forwaard(s_rep, acts)
            q_2 = self.critic_target.forwaard(s_rep, acts)
            q = T.min(q_1, q_2)
            q = q.view(-1)

            actor_loss = probs - q
            actor_loss = T.mean(actor_loss)
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            # critic loss
            self.critic.optimizer.zero_grad()
            self.critic_target.optimizer.zero_grad()
            q_est = self.reward_scale*r_rep + self.gamma*value_1
            q1_crit = self.critic.forwaard(s_rep, a_rep).view(-1)
            q2_crit = self.critic_target.forwaard(s_rep, a_rep).view(-1) 
            critic_loss_1 = .5 * F.mse_loss(q1_crit, q_est)
            critic_loss_targ = .5 * F.mse_loss(q2_crit, q_est)
            
            critic_loss = critic_loss_1 + critic_loss_targ
            critic_loss.backward()
            self.critic.optimizer.step()
            self.critic_target.optimizer.step()

            # update targets
            self.update_target()


    def calculate_q(self, s_rep, reparam=False):
            acts, probs = self.actor.normaalize_sample(s_rep,reparam)
            probs = probs.view(-1)
            q_1 = self.critic.forwaard(s_rep, acts)
            q_2 = self.critic_target.forwaard(s_rep, acts)
            q = T.min(q_1, q_2)
            q = q.view(-1)
            return probs, q


    def update_target(self, tau=.001):
        value = self.value.named_parameters()
        value_target_params = self.value_target.named_parameters()        
        value_dict = dict(value)
        value_targ_dict = dict(value_target_params)
        
        for name in value_dict:
            value_dict[name] = tau*value_dict[name].clone() +\
                                (1-tau)*value_targ_dict[name].clone()

        self.value_target.load_state_dict(value_dict)

    def save_model(self,dir):
        self.actor.save_model(dir)
        self.critic.save_model(dir)
        self.value.save_model(dir)

