#Balancing the pole
#195 frames to be held (die if fall off)
#

#Select GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Import libraries
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Set Parameters
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
n_episodes = 1001
output_dir = 'model_output/cartpole'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Define agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        #Use random samples because similar events do not give much information

        #2000개까지만 저장해줌
        self.memory = deque(maxlen=2000)

        #Gamma parameter for discounting the reward
        self.gamma = 0.96

        #Exploration vs. Exploitation
        #확률에 따라 explore을 하지 exploit을 하지 정하는 거
        #epsilon 1이면 처음에는 explore만 하도록 하는 거
        self.epsilon = 1

        #Decay를 해주면 exploration에서 exploitation으로 옮겨간다
        #Min을 설정해주면 decay가 끝나고 최소 0.01로 남는다
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.learning_rate = 0.001

        #Private method: means that this method can only be used by this specific instance of the class
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        #First layer
        model.add(Dense(24, input_dim = self.state_size, activation = 'relu'))
        #Second layer (shallow한 layer이다)
        model.add(Dense(24, activation = 'relu'))
        #Linear -> not want any abstract actions; want direct ones
        model.add(Dense(self.action_size, activation = 'linear'))

        #Loss계산 방법과 optimizer의 기법 정하기
        model.compile(loss='mse', optimizer = Adam(lr = self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #If smaller than epsilon than exploration mode
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #If not than exploitation mode
        #For exploitation pass in state information and use the predict method
        #This lets predict the best course of action
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        #Randomly sample from memory
        #batch_size -> the number of samples
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # Maximum number of steps in the game is 200 -> done
            # If failed -> done
            # If done then target = reward
            target = reward

            #If not done than needs prediction of the future reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)

            #action에 대한 prediction 구하기
            target_f[0][action] = target

            #Fitting하기
            self.model.fit(state, target_f, epochs = 1, verbose = 0)

        #Epsilon decay 시키기
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self,name):
        self.model.save_weights(name)

#Defining the agent
agent = DQNAgent(state_size, action_size)

#Interacting with the environment
done = False
for e in range(n_episodes):
    #Starting each episode with resets
    state = env.reset()
    #Neural Network에 맞게 resize
    state = np.reshape(state, [1,state_size])

    #If does not fall, 5000 is the maximum number of time steps
    #195 frame을 reach 한다고 그냥 끝나는 게 아니다?
    for time in range(5000):
        #Showing the env on screen
        env.render()

        #Executing action
        action = agent.act(state)

        #Action에 따른 값들 받기
        next_state, reward, done, _ = env.step(action)

        #Reward = reward (if fail than -10)
        reward = reward if not done else -10

        next_state = np.reshape(next_state, [1, state_size])

        #Put into remember
        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("Episode: {}/{}, Score: {}, Epsilon: {:.2}".format(e, n_episodes, time, agent.epsilon))
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")






























