import gym, assistive_gym

env = gym.make('JointReachingPR2-v1')
env.render()
observation = env.reset()

while True:
    env.render()
    action = env.action_space.sample() # Get a random action
    #print('action: ',action) PR2 has 7 actions
    observation, reward, done, info = env.step(action)
