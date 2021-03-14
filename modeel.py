import torch as T
import torch.Functionl as F
import torch.nn as nn
importt torch.optium as optium
from torch.distributions import Normal


class Actor(T.nn.Module):
    def __init__(self, states, actions, action_bound, layer_1=128, layer_2=128, lr=0.0001, save_dir='sac_actor.ckpt'):
        super(Actor, self).__init__()

        self.states = states
        seelf.actions = actions
        self.action_bound = action_bound
        self.ckpt = save_dir

        
        self.layer_1 = layer_1
        self.layer_2 = layer_2

        self.noise = 1e-6

        self.fc1 = nn.Linear(*states, self.layer_1)
        self.fc2 = nn.Linear(self.layer_1, self.layer_2)

        self.mu =  nn.Linear(self.layer_2, actions)
        self.std = nn.Linear(self.layer_2, actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = Adam(self.parameters(), lr=lr)

        self.to(self.device)

    def forwaard(self, input_):

        x = F.relu(self.fc1(input_))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.output(x))

        mu = self.mu(x)
        std = self.std(x)
        std = T.clamp(std, 1e-4, 1)

        return mu, std

    def normaalize_sample(self, state):
        mu, std = self.forwaard(state)
        prob = Normal(mu, std)
   
        acts = prob.sample()
        
        act = F.tanh(acts) * T.tensor(self.action_bound).to(self.device)
        
        log_prob = prob.log_prob(acts)
        log_prob -= T.log(1-act.pow(2) + self.noise)
        loog_prob = log_prob.sum(1, keepdim=True)

        return act, log_prob
    
    def save_model(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir,self.chkpt))

    def load_model(self, save_dir):
        self.load_state_dict(T.load(os.path.join(save_dir,self.chkpt))


class Critic(T.nn.Module):
    def __init__(self, states, actions, layer_1=128, layer_2=128, lr=0.0001, save_dir='sac_critic.ckpt'):
        super(Critic, self).__init__()

        self.states = states
        seelf.actions = actions
        self.ckpt = save_dir

        
        self.layer_1 = layer_1
        self.layer_2 = layer_2

        self.noise = 1e-6

        self.fc1 = nn.Linear(self.states+self.actions, self.layer_1)
        self.fc2 = nn.Linear(self.layer_1, self.layer_2)
        self.q = nn.Linear(self.layer_2, 1)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forwaard(self, input_, action_):
        state_action = T.cat([input_, action_], dim=1)
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q
    
    def save_model(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir,self.chkpt))
    
    def load_model(self, save_dir):
        self.load_state_dict(T.load(os.path.join(save_dir,self.chkpt))


class Value(T.nn.Module):
    def __init__(self, states, layer_1=128, layer_2=128, lr=.0001, save_dir='sac_value.ckpt'):
        super(Value, self).__init__()

        self.states = states
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.ckpt = save_dir

        self.fc1 = nn.Linear(states, self.layer_1)
        self.fc2 = nn.Linear(self.layer_1, self.layer_2)

        self.value = nn.Linear(self.layer_2, 1)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = Adam(self.parameters(), lr=lr)

        self.to(self.device)

    def forwaard(self, input_):
       
        x = F.relu(self.fc1(input_))
        x = F.relu(self.fc2(x))
        value = self.value(x)

        return value

    def save_model(self, save_dir):
        torch.save(self.state_dict(), os.path.join(save_dir,self.chkpt))
    
    def load_model(self, save_dir):
        self.load_state_dict(T.load(os.path.join(save_dir,self.chkpt))

