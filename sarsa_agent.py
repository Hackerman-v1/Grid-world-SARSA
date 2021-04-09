# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:11:09 2021

@author: Ramesh
"""

from GridWrld_env import GridWorld
import matplotlib.pyplot as plt
import torch
import numpy as np

""" Qvalu is the table to store the Qvalue for each state action based on the Q-formulae
    it is very very necessary to intiaite this variable Globally because the table should 
    not be updated to zeros, instead it has to update the Q-value for the right predictions.
    """
Q_valu = torch.zeros(2500,4) 


class Q_learning :
    def __init__(self, epsilon, lr_rate, reward_decay):
        self.epsilon = epsilon
        self.Q_val = Q_valu
        self.lr = lr_rate
        self.gamma = reward_decay
        self.actionpossibilities = ['U', 'D', 'L' , 'R'] #possible actions 
        
    def take_action(self, state): #taking actions based on tbe E-greedy method
        
        if np.random.uniform()<self.epsilon:
            action = np.random.choice(self.actionpossibilities)
        else:
            action = int(torch.argmax(self.Q_val[state,:]))
            action = self.actionpossibilities[action]
        return action
            
    def Q_upgrade(self,state,action,rew,next_state, next_action): # updating the Q-value table
        prediction = self.Q_val[state, action]
        
        target = rew + self.gamma*self.Q_val[next_state, next_action]
        
        self.Q_val[state,action] = prediction + self.lr*(target-prediction)
        
        
if __name__ == '__main__':
    num_episode = 1000
    Q_value = torch.zeros([num_episode,1])
    env = GridWorld(50,50)
    c_qlearn = Q_learning(0.9, 0.01, 0.85) 
    possibilities =['U', 'D', 'L' , 'R']
    rewards = torch.zeros([num_episode,1])
    step_size = torch.zeros([num_episode,1])
    for episode in range(num_episode):
        obs = int(env.reset())
        done = False
        sum_rew = 0
        steps = 0
        while True:
            action = c_qlearn.take_action(obs)
            next_state, rew, done, info = env.step(action)
            for i in range (len(possibilities)):
                if action == possibilities[i]:
                    action  = i
            next_action = c_qlearn.take_action(next_state)
            for i in range (len(possibilities)):
                if next_action== possibilities[i]:
                    next_action = i
            next_state = int(next_state)
            c_qlearn.Q_upgrade(obs, action, rew, next_state, next_action)
            obs = next_state
            sum_rew += rew
            steps+=1
            if done:
                break
        print(episode)   
        Q_value[episode] = torch.mean(c_qlearn.Q_val)
        rewards[episode] = sum_rew  
        step_size[episode] = steps

    plt.plot(rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    
    plt.plot(step_size)  
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Number of steps')
    plt.show()

    plt.plot(Q_value)
    plt.xlabel('Qvalue')
    plt.ylabel('Episode')
    plt.plot()
    plt.show()
            
            
            
        
    