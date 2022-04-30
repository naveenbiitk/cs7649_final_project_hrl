import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

import os
import ray._private.utils
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from assistive_gym.learn import make_env

env_name = "JointMotionStretchHuman-v1"
env = make_env(env_name, coop=True)
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

batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
writer = JsonWriter(
    os.path.join(os.getcwd(), "demo-out")
)

# print("OBS:", env.observation_space)

prep = get_preprocessor(env.observation_space)(env.observation_space)
print("The preprocessor is", prep)

for eps_id in range(1):
    obs = env.reset()
    print("OBS: ", obs)
    obs= np.concatenate((obs['robot'], obs['human']))
    print("OBS: ", obs)

    prev_action = np.zeros_like(env.action_space.sample())
    prev_reward = 0
    done = False
    t = 0
    while (not done) and (t < 150):
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

        final_action =  {'robot': robot_action*100, 'human': human_action*20}

        # observation, reward, done, info = env.step(action*100)

        new_obs, rew, done, info = env.step(final_action)
        print("NEW_OBS: ", new_obs)
        new_obs=np.concatenate((new_obs['robot'], new_obs['human']))
        print("NEW_OBS: ", new_obs)


        batch_builder.add_values(
            t=t,
            eps_id=eps_id,
            agent_index=0,
            # obs=prep.transform(np.concatenate(obs['robot'], obs['human'])),
            obs=prep.transform(obs),
            actions=final_action,
            action_prob=1.0,  # put the true action probability here
            action_logp=0.0,
            rewards=rew,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            dones=done,
            infos=info,
            # new_obs=prep.transform(np.concatenate(new_obs['robot'], new_obs['human'])),
            new_obs=prep.transform(new_obs),
            
        )
        # print(obs)
        obs = new_obs
        prev_action = final_action
        prev_reward = rew
        t += 1
        print(t)

    writer.write(batch_builder.build_and_reset())
