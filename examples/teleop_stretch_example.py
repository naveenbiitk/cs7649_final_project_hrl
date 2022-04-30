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
robot_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

# human_actions = { ord('7'): np.array([ 0.01, 0, 0, 0]), ord('8'): np.array([ -0.01, 0, 0, 0]), ord('u'): np.array([ 0, 0.01, 0, 0]), ord('i'): np.array([ 0, -0.01, 0, 0]), ord('j'): np.array([ 0, 0, -0.01, 0]), ord('k'): np.array([ 0, 0, 0.01, 0]), ord('n'): np.array([ 0, 0, 0, 0.01]), ord('m'): np.array([0, 0, 0, -0.01])}
# human_actions = {ord('1'): np.array([0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#                 ord('e'): np.array([-0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#                 ord('2'): np.array([0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0]),
#                 ord('r'): np.array([0, -0.01, 0, 0, 0, 0, 0, 0, 0, 0]),
#                 ord('3'): np.array([0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0]),
#                 ord('t'): np.array([0, 0, -0.01, 0, 0, 0, 0, 0, 0, 0]),
#                 ord('4'): np.array([0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0]),
#                 ord('y'): np.array([0, 0, 0, -0.01, 0, 0, 0, 0, 0, 0]),
#                 ord('5'): np.array([0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0]),
#                 ord('u'): np.array([0, 0, 0, 0, -0.01, 0, 0, 0, 0, 0]),
#                 ord('6'): np.array([0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0]),
#                 ord('i'): np.array([0, 0, 0, 0, 0, -0.01, 0, 0, 0, 0]),
#                 ord('7'): np.array([0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0]),
#                 ord('o'): np.array([0, 0, 0, 0, 0, 0, -0.01, 0, 0, 0]),
#                 ord('8'): np.array([0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0]),
#                 ord('p'): np.array([0, 0, 0, 0, 0, 0, 0, -0.01, 0, 0]),
#                 ord('9'): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0]),
#                 ord('['): np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.01, 0]),
#                 ord('0'): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01]),
#                 ord(']'): np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -0.01])
# }

# start
right_arm_angles_start = np.array([0.0,  0.0,  5.31665415e-01,  1.34692067e-01, -9.64689935e-02, -2.42673015e-01, -1.64185327e+00,  0.0,  0.0,  0.0])

#end
# right_arm_angles_end = np.array([0.0,  0.0, 0.0,  0.6285238,  -1.2184946,   0.0,  -2.2070327,  0.0, 0.0, 0.0])
right_arm_angles_end = np.array([0.0,  0.0, 0.0,  0.6285238 - 0.5,  -1.2184946,   0.0 + 0.25,  -2.2070327,  0.0, 0.0, 0.0])

delta = (right_arm_angles_end - right_arm_angles_start) / 100

# wrong = np.array([-0.03054897,  0.02017162, -0.49140823,  1.0715723,  -1.13699922,  0.81717021, -2.21021678,  0.00261914, 0.20373329,  0.14336391])

# print("ERROR:", right_arm_angles_end - wrong)

human_actions = {ord('1'): delta, ord('2'): -delta}


while True:
    env.render()

    human_action = np.zeros(env.action_human_len)
    keys = p.getKeyboardEvents()
    for key, a in human_actions.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            human_action += a

    robot_action = np.zeros(env.action_robot_len)
    keys = p.getKeyboardEvents()
    for key, a in robot_actions.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            robot_action += a

    if coop:
        final_action =  {'robot': robot_action*100, 'human': human_action*20}
    else:
        final_action = robot_action

    observation, reward, done, info = env.step(final_action)

