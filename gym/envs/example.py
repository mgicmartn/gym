import gym
env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=42, return_info=True)
episodes = 10
for episode in range(1, episodes + 1):
    state =env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()  
        observation, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    ##if done:
      ##observation, info = env.reset(return_info=True)
env.close()