import gym, assistive_gym
from assistive_gym.learn import make_env

# env = gym.make('JointMotionStretchHuman-v1')
env_name = 'JointMotionStretchHuman-v1'
env = make_env(env_name, coop=True)
env.render()
observation = env.reset()

action = dict()

while True:
    env.render()
    action['robot'] = env.action_space_robot.sample() # Get a random action
    action['human'] = env.action_space_human.sample() # Get a random action
    #print('action: ',action) PR2 has 7 actions
    observation, reward, done, info = env.step(action)
