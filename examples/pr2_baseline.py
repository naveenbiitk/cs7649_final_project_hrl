import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import sys
import math


def evaluate_baseline(seed):

    n_episodes = 100
    env = gym.make(args.env)
    env.seed(seed)
    #env.render()
    rewards_ls = []
    forces_ls = []
    task_successes_ls = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        force_list = []
        task_success = 0.0
        pos, orient = env.robot.get_pos_orient(env.robot.left_end_effector)
        human_target = np.array(env.target_pos)

        target_pose = env.target_pos
        human_target_2 = [human_target[0]-0.15, human_target[1], human_target[2]+0.15 ]
        target_pose_angles_1 = env.robot.ik(env.robot.left_end_effector, human_target_2, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
        target_pose_angles_2 = env.robot.ik(env.robot.left_end_effector, human_target, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)

        #print('robot target angle',target_pose_angles_1)

        current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

        #print('current_joint_angles of robot',current_joint_angles)

        #############################################################

        count_= 0
        reward_total = 0
    
        while count_<200:
            #env.render()
            action = np.zeros(env.action_robot_len)
            # if count_<50:
            #     base_position, _, _ = env.robot.position_robot_toc(env.task, 'left', pos, target_pose, env.human, step_sim=False, check_env_collisions=True, max_ik_iterations=100, max_ik_random_restarts=1, randomize_limits=False, right_side=right_side, base_euler_orient=[0, 0, 0 if right_side else np.pi], attempts=50)
            #print('inside loop')
            #current_joint_angles = env.robot.get_joint_angles(env.robot.left_gripper_indices)
            #action[len(env.robot.wheel_joint_indices):] = target_pose_angles-current_joint_angles
        
            current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
        
            if count_<80:
                action = target_pose_angles_1-current_joint_angles
            else:
                action = target_pose_angles_2-current_joint_angles

            #print('count: ',count_,' Actions: ',action)
        
            count_ = count_+1
            observation, reward, done, info = env.step(action)
            reward_total += reward
            task_success = info['task_success']


        ##############################################################

        print('Episode:',episode,'   reward: ',reward_total,'  task_success:',task_success)
        rewards_ls.append(reward_total)
        forces_ls.append(np.mean(force_list))
        task_successes_ls.append(task_success)
        print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))


    env.disconnect()

    print('\n', '-'*50, '\n')
    print('Number of episodes', n_episodes)
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards_ls))
    print('Reward Std:', np.std(rewards_ls))

    # print('Forces:', forces)
    print('Force Mean:', np.mean(forces_ls))
    print('Force Std:', np.std(forces_ls))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes_ls))
    print('Task Success Std:', np.std(task_successes_ls))
    print('Task_successes total', np.sum(task_successes_ls))
    print('Task_successes total', task_successes_ls)
    sys.stdout.flush()



def run_baseline():
    print('--------------------------------------')
    env = gym.make(args.env)
    seed=2
    env.seed(seed)
    env.render()
    observation = env.reset()
    #env.robot.print_joint_info()

    print('--------------------------------------')
    print('human limmb pose',env.human.get_pos_orient(env.limb))

    pos, orient = env.robot.get_pos_orient(env.robot.left_end_effector)

    #print(pos)
    #print(orient)
    human_target = np.array(env.target_pos)
    print('target pose',human_target)
    #-0.36859864 -0.06440637  0.80693936  1.
    #-0.36859864 -0.06440637  0.80693936  2.

    #human_target[0]=-0.3859
    #human_target[1]=0.46769
    #human_target[2]=0.849474
    print('--------------------------------------')
    # Arrow keys for moving the base, 
    # s/x for the lift, third array
    # z/c for the prismatic joint, fourth array (compression/retraction)
    # a/d for the wrist joint, fifth array  (yaw movements)

    #keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

    target_pose = env.target_pos
    human_target_2 = [human_target[0]-0.2, human_target[1], human_target[2]+0.4 ]
    target_pose_angles_1 = env.robot.ik(env.robot.left_end_effector, human_target_2, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
    target_pose_angles_2 = env.robot.ik(env.robot.left_end_effector, human_target, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)

    print('robot target angle',target_pose_angles_1)

    current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

    print('current_joint_angles of robot',current_joint_angles)

    action = np.zeros(env.action_robot_len)
    right_side=False
    count_= 0

    reward_total = 0
    while count_<200:
        env.render()
        action = np.zeros(env.action_robot_len)
        # if count_<50:
        #     base_position, _, _ = env.robot.position_robot_toc(env.task, 'left', pos, target_pose, env.human, step_sim=False, check_env_collisions=True, max_ik_iterations=100, max_ik_random_restarts=1, randomize_limits=False, right_side=right_side, base_euler_orient=[0, 0, 0 if right_side else np.pi], attempts=50)
        #print('inside loop')
        #current_joint_angles = env.robot.get_joint_angles(env.robot.left_gripper_indices)
        #action[len(env.robot.wheel_joint_indices):] = target_pose_angles-current_joint_angles
        
        current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
        
        if count_<80:
            action = target_pose_angles_1-current_joint_angles
        else:
            action = target_pose_angles_2-current_joint_angles

        print('count: ',count_,' Actions: ',action)
        
        count_ = count_+1
        observation, reward, done, info = env.step(action)
        reward_total += reward
        task_success = info['task_success']


    print('reward')
    print(reward_total)
    print('task_success')
    print(task_success)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='JointReachingPR2-v1',
                    help='Environment to test (default: JointReachingStretchHuman-v1)')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--seed', type=int, default=5,
                        help='Random seed (default: 5)')
    args = parser.parse_args()
    #run_baseline()
    dump = np.array([1,2,3,4,5,15])
    for i_ in range(5):
        print('Big for loop ',i_)
        evaluate_baseline(seed=int(dump[i_]))

    #evaluate_baseline()