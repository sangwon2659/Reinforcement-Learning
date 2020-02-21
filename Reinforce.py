# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:52:27 2020

@author: 283
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    avg_t = 0
    
    for n_epi in range(500):
        obs = env.reset()
        done = False
        for t in range(600):
            obs = torch.tensor(obs,dtype = torch.float) #tensor로 이루어진 observation
            out = pi(obs) #policy에 넣어준 결과를 out이라는 변수로 저장
            m = Categorical(out) #pytorch에서 지원하는 확률분포 모델
            action = m.sample() #m에서 action pick하기
            
            if avg_t/30.0>200:
                env.render()
            
            obs, r, done, infor = env.step(action.item()) #tensor인 action을 scalar로 바꿔서 넣어주면 obs, r, done이 나온다
            pi.put_data((r,torch.log(out[action]))) #policy안에 r과 log pi값 넣어놓기
            if done:
                break
        avg_t += t
        pi.train()
        if n_epi%20 == 0 and n_epi != 0:
            print('# of episode: {}, Avg timestep: {}'.format(n_epi, avg_t/20.0))
            avg_t = 0
    env.close()
    
class Policy(nn.Module): #nn.Module이라는 pytorch에 있는 class를 상속 받아서 선언
    def __init__(self):
        super(Policy, self).__init__() #상위 클래스에서 받아오기
        self.data = [] #data라는 list 만들어주기
        self.gamma = 0.99 #gamma 상수 선언하기
        
        self.fc1 = nn.Linear(4,128) #Layer 쌓기
        self.fc2 = nn.Linear(128,2)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005) #Adam 방식으로 optimize
        
    def forward(self,x): #아마 내장되어 있는 함수에서 사용하는 함수
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0) #확률에서 값 추출하기
        print('a')
        return x
    
    def put_data(self, item):
        self.data.append(item) #self.data에 값 넣어주기
    
    def train(self):
        R = 0 #Reward setting
        for r, log_prob in self.data[::-1]: #뒤에서부터 self.data 읽기
            R = r + R * self.gamma #decay term인 gamma 계속 곱해주기
            loss = -log_prob * R #loss 계산
            self.optimizer.zero_grad() #내장 함수
            loss.backward() #내장 함수
            self.optimizer.step() #알아서 update 해줌
        self.data = [] #self.data 초기화
    
if __name__ == '__main__':
    main()
            