import gym, assistive_gym, argparse
import pybullet as p
import numpy as np

import os
import ray._private.utils
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
parser.add_argument('--env', default='JointMotionStretch-v1',
                    help='Environment to test (default: ScratchItchStretch-v1)')
args = parser.parse_args()

env = gym.make(args.env)
env.render()
observation = env.reset()
# env.robot.print_joint_info()

# Arrow keys for moving the base, s/x for the lift, z/c for the prismatic joint, a/d for the wrist joint
keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
writer = JsonWriter(
    os.path.join(os.getcwd(), "demo-out")
)

# print("OBS:", env.observation_space)

prep = get_preprocessor(env.observation_space)(env.observation_space)
print("The preprocessor is", prep)

for eps_id in range(1):
    obs = env.reset()
    prev_action = np.zeros_like(env.action_space.sample())
    prev_reward = 0
    done = False
    t = 0
    while (not done) and (t < 150):
        env.render()
        action = np.zeros(env.action_robot_len)
        keys = p.getKeyboardEvents()
        for key, a in keys_actions.items():
            if key in keys and keys[key] & p.KEY_IS_DOWN:
                action += a

        # observation, reward, done, info = env.step(action*100)

        new_obs, rew, done, info = env.step(action*100)
        batch_builder.add_values(
            t=t,
            eps_id=eps_id,
            agent_index=0,
            obs=prep.transform(obs),
            actions=action,
            action_prob=1.0,  # put the true action probability here
            action_logp=0.0,
            rewards=rew,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            dones=done,
            infos=info,
            new_obs=prep.transform(new_obs),
        )
        # print(obs)
        obs = new_obs
        prev_action = action
        prev_reward = rew
        t += 1
        print(t)

    writer.write(batch_builder.build_and_reset())
