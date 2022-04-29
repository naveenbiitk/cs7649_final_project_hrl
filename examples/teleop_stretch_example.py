import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
from assistive_gym.learn import make_env
#parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
#parser.add_argument('--env', default='JointMotionStretchHuman-v1',
#                    help='Environment to test (default: ScratchItchStretch-v1)')
#args = parser.parse_args()
#env_name = "JointMotionStretch-v1"
env_name = "JointMotionStretchHuman-v1"
coop = 'Human' in env_name
env = make_env(env_name, coop=True) if coop else gym.make(env_name)
#env = gym.make()
env.render()
observation = env.reset()
env.robot.print_joint_info()

# Arrow keys for moving the base, s/x for the lift, z/c for the prismatic joint, a/d for the wrist joint
#keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

keys_actions = { ord('7'): np.array([ 0.01, 0, 0, 0]), ord('8'): np.array([ -0.01, 0, 0, 0]), ord('u'): np.array([ 0, 0.01, 0, 0]), ord('i'): np.array([ 0, -0.01, 0, 0]), ord('j'): np.array([ 0, 0, -0.01, 0]), ord('k'): np.array([ 0, 0, 0.01, 0]), ord('n'): np.array([ 0, 0, 0, 0.01]), ord('m'): np.array([0, 0, 0, -0.01])}

while True:
    env.render()

    #action = np.zeros(env.action_robot_len)
    action = np.array([0.0,0.0,0.0, 0.0])
    keys = p.getKeyboardEvents()
    for key, a in keys_actions.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            action += a

    if coop:
        final_action =  {'robot': np.array([0,0,0,0,0]), 'human': action*20 }
        #print('Human action',action)
    else:
        final_action = action

    observation, reward, done, info = env.step(final_action)

