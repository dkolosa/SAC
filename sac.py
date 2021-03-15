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

        self.memory = Uniform_Memory(buffer_size=100000)

        self.update_target(self.tau)

    
    def action(self, state):
        self.actor.train()
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        act, _ = self.actor.forwaard(state)
        self.actor.eval()
        return act.cpu().detach().numpy()[0]

    def train(self):
        if self.batch_size < self.memory.get_count:
            mem = self.memory.sample(self.batch_size)
            s_rep = T.tensor(np.array([_[0] for _ in mem]), dtype=T.float).to(self.actor.device)
            a_rep = T.tensor(np.array([_[1] for _ in mem]), dtype=T.float).to(self.actor.device)
            r_rep = T.tensor(np.array([_[2] for _ in mem]), dtype=T.float).to(self.actor.device)
            s1_rep = T.tensor(np.array([_[3] for _ in mem]), dtype=T.float).to(self.actor.device)
            d_rep = T.tensor(np.array([_[4] for _ in mem]), dtype=T.float).to(self.actor.device)

            self.critic.eval()
            self.actor.eval()
            self.critic_target.eval()
            self.value.eval()
            self.value_target.eval()

            # Calculate critic and train
            value = self.value.forwaard(s_rep).view(-1)
            value_1 = self.value.forwaard(s1_rep).view(-1)

            acts, probs = self.actor.normaalize_sample(s1_rep,reparam=False)
            probs = probs.view(-1)
            q_1 = self.critic.forwaard(s1_rep, acts)
            q_2 = self.critic_target.forwaard(s1_rep, acts)
            q = T.min(q_1, q_2)
            q = q.view(-1)

            # Value network loss
            self.value.train()
            self.value.optimizer.zero_grad()
            value_target = q - probs
            value_loss = .5 * F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()
            self.value.eval()

            # actor loss
            acts, probs = self.actor.normaalize_sample(s1_rep, reparam=True)
            probs = probs.view(-1)
            q_1_act = self.critic.forwaard(s1_rep, acts)
            q_2_act = self.critic_target.forwaard(s1_rep, acts)
            q_act = T.min(q_1_act, q_2_act)
            q_act = q_act.view(-1)

            self.actor.train()
            actor_loss = probs - q_act
            actor_loss = T.mean(actor_loss)
            self.actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()
            self.actor.eval()

            # critic loss
            self.critic.train()
            self.critic_target.train()
            self.critic.optimizer.zero_grad()
            self.critic_target.optimizer.zero_grad()
            q_est = r_rep + self.gammaa*value
            q1 = self.critic.forwaard(s_rep, a_rep).view(-1)
            q2 = self.critic_target.forwaard(s_rep, a_rep).view(-1) 
            critic_loss = .5 * F.mse_loss(q1, q_est)
            critic_loss_targ = .5 * F.mse_loss(q2, q_est)
            
            critic_loss = critic_loss + critic_loss_targ
            critic_loss.backward()
            self.critic.optimizer.step()
            self.critic_target.optimizer.step()

            self.update_target()



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

