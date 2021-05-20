import torch
import torch.nn as nn
import random
import numpy as np



class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_action):
        super(Actor, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.max_action = max_action
    def forward(self, x):
        x = self.h1(x)
        x = self.relu(x)
        x = self.h2(x)
        x = self.tanh(x)*self.max_action
        return x




class Critic(nn.Module):
    def __init__(self, state_input_size, action_input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.h1 = nn.Linear(state_input_size +  action_input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        x = self.h1(x)
        x = self.relu(x) 
        x = self.h2(x)
        
        return x


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.idx     = 0
        self.pointer = 0
        self.state_buffer       = np.zeros((capacity, dims))  
        self.action_buffer      = np.zeros((capacity, 1))
        self.reward_buffer      = np.zeros((capacity, 1))
        self.next_state_buffer  = np.zeros((capacity, dims))  
        

    def __len__(self):
        return len(self.state_buffer)

    def store_transition(self, state, action, reward, next_state):
        
        self.idx %= self.capacity

        state = state.reshape(-1)
        action = action.reshape(-1)
        reward = reward
        next_state = next_state.reshape(-1)
        
    
        self.state_buffer[self.idx]         = state       
        self.action_buffer[self.idx]        = action      
        self.reward_buffer[self.idx]        = reward      
        self.next_state_buffer[self.idx]    = next_state  

        self.idx +=1
        self.pointer+=1


        # if(len(self.state_buffer) > self.capacity):
        #     self.state_buffer       = self.state_buffer[1:]
        #     self.action_buffer      = self.action_buffer[1:]
        #     self.reward_buffer      = self.reward_buffer[1:]
        #     self.next_state_buffer  = self.next_state_buffer[1:]
        
        
    def sample(self, n):
        idx =  np.random.choice(self.capacity, size=n)
        return np.array(self.state_buffer)[idx], np.array(self.action_buffer)[idx],\
             np.array(self.reward_buffer)[idx], np.array(self.next_state_buffer)[idx]