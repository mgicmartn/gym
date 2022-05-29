# Imports
import io
import os
import glob
import torch
import base64
import stable_baselines3
import time

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gym
from gym import spaces
from gym.wrappers import Monitor

nn_layers = [64,64] #configuration of the neural network.
                    

learning_rate = 0.001

#log_dir = "/tmp/gym/"
#os.makedirs(log_dir, exist_ok=True)

# Create environment
env = gym.make('LunarLanderContinuous-v2')

#env = stable_baselines3.common.monitor.Monitor(env, log_dir )
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, verbose=1, learning_rate = 0.1)
model.learn(total_timesteps=100000, log_interval=10)

#env = model.get_env()
model.save("ddpg_pendulum")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_pendulum")


"""
callback = EvalCallback(env,log_path = log_dir, deterministic=True) 
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)
model = DDPG("MlpPolicy", env,policy_kwargs = policy_kwargs,
            learning_rate=learning_rate,
            batch_size=1,  #for simplicity, we are not doing batch update.
            buffer_size=1, #size of experience of replay buffer. Set to 1 as batch update is not done
            learning_starts=1, #learning starts immediately!
            gamma=0.80, #discount facto. range is between 0 and 1.
            tau = 1,  #the soft update coefficient for updating the target network
            
            train_freq=(1,"step"), #train the network at every step.
            
            gradient_steps = 1, #number of gradient steps
            seed = 1, #seed for the pseudo random generators
            verbose=1) #Set verbose to 1 to observe training logs. We encourage you to set the verbose to 1.

model.learn(total_timesteps=100000, log_interval=10, callback=callback)


#env = gym.make("LunarLander-v2")
#observation = env.reset()

while True:
  env.render()
  action, _states = model.predict(observation, deterministic=True)
  observation, reward, done, info = env.step(action)
  if done:
    break;
"""
episodes = 5
for episode in range(1, episodes + 1):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        time.sleep(.02)
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    
env.close()
env.close()
