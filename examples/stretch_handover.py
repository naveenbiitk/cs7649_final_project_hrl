import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import sys
import math


def evaluate_baseline():
    seed=5

    n_episodes = 5
    env = gym.make(args.env)
    env.seed(seed)
    env.render()
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
        # human_pos_real, _ = env.robot.convert_to_realworld(env.target_pos)
        #print('difference', human_pos_real-human_target)
        target_pose_angles = env.robot.ik(env.robot.left_end_effector, human_target, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
        action = np.zeros(env.action_robot_len)
        count_= 0
        tool_pos, tool_orient = env.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = env.robot.convert_to_realworld(tool_pos, tool_orient)

        while not done:
            env.render()

#########To check directions
            base_pose,base_orient = env.robot.get_pos_orient(env.robot.base)
            #print('Direction: ',math.atan2(base_pose[1]-human_target[1],base_pose[0]-human_target[0])/math.pi*180)

###########
            action = np.zeros(env.action_robot_len)
            current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
            target_pose_angles = env.robot.ik(env.robot.left_end_effector, human_target, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
            #current_joint_angles = env.robot.get_joint_angles(env.robot.left_gripper_indices)
            action[len(env.robot.wheel_joint_indices):] = target_pose_angles-current_joint_angles
              
            y_offset=-0.1

            if count_<50:
                action[0] = human_target[1]-base_pose[1]+y_offset
                action[1] = action[0]
                #action[2] = -0.5*(target_pose_angles[0]-current_joint_angles[0])
                action[2]=0
            else:
                action[2] = (target_pose_angles[0]-current_joint_angles[0])+0.02

            action[0] = human_target[1]-base_pose[1]+y_offset
            action[1] = action[0]
            action[3] = (target_pose_angles[1]-current_joint_angles[1])
            action[4] = (target_pose_angles[2]-current_joint_angles[2])-0.15
            
            pos, orient_1 = env.robot.get_pos_orient(env.robot.left_end_effector)

            #print("Actions ",np.linalg.norm(action), "Poses",np.linalg.norm(pos-human_target))
            count_ = count_ + 1

            # if np.linalg.norm(action)<0.04 and count_>90:
            #     action[2]=0
            #     action[3]=0
            #     action[4]=0

            action = action*1
            observation, reward, done, info = env.step(action)
            reward_total += reward
            task_success = info['task_success']
            force_list.append(info['total_force_on_human'])

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


#angular pid constansts
prev_angle_term = 0
integral_object_angle = 0
#linear pid constants
prev_linear_term = 0
integral_linear_gap = 0


def linear_controller(object_gap):
    kp_gap = 4.25/1.3
    kd_gap = 0.04
    ki_gap = 0.0000
    
    global prev_linear_term,integral_linear_gap
    z = 0
    dt = 0.1

    #object_gap = setpoint_x-current_x
    
    derivative_object_gap = (object_gap - prev_linear_term)/dt
    integral_object_gap = integral_linear_gap+object_gap*dt

    z = kp_gap*object_gap + kd_gap*derivative_object_gap + ki_gap*integral_object_gap #Remove if object_gap < 0 statement and replace with this line

    prev_linear_term = object_gap
    integral_linear_gap = integral_object_gap

    return z
    
    
    
def angular_controller(object_angle_gap):
    kp_angle_gap = 4.02/2.5
    kd_angle_gap = 0.00#5
    ki_angle_gap = 0.0000

    global prev_angle_term,integral_object_angle
    z = 0
    dt = 0.1

    #object_angle_gap = setpoint_theta-current_theta
    #object_angle_gap = min(object_angle_gap, 360-object_angle_gap)

    derivative_object_angle_gap = (object_angle_gap - prev_angle_term)/dt
    integral_object_angle_gap = integral_object_angle +object_angle_gap*dt

    z = kp_angle_gap*object_angle_gap + kd_angle_gap*derivative_object_angle_gap + ki_angle_gap*integral_object_angle_gap

    prev_angle_term = object_angle_gap
    integral_object_angle = integral_object_angle_gap
    #print('Ki angle term',ki_angle_gap*integral_object_angle_gap)
    return z



def position_controller(setpoint_position, current_position):

    error_position = np.linalg.norm(np.array(setpoint_position[:2])-np.array(current_position[:2]))
    error_theta = (np.linalg.norm(np.array(setpoint_position[2])-np.array(current_position[2])))%360
    linear_v = 0.0
    angular_v = 0.0
    error_threshold = 0.32
    theta_threshold = 2
    position_threshold = 0.2

    if abs(error_position) > position_threshold or min(abs(error_theta),360-abs(error_theta)) > theta_threshold:
        if abs(error_position) > position_threshold:#position control
            theta_del = math.atan( (np.array(setpoint_position[1])-np.array(current_position[1]))/(np.array(setpoint_position[0])-np.array(current_position[0])) )
            d_del = np.max(np.abs(np.array(setpoint_position[:2])-np.array(current_position[:2])))

            linear_v = linear_controller(d_del)
            angular_v = angular_controller(theta_del)
            #print(' Position change: ', error_position, linear_v)
        elif min(abs(error_theta),360-abs(error_theta)) > theta_threshold:
            print('Angular control')
            angular_v = angular_controller(np.array(setpoint_position[2])-np.array(current_position[2]))
            angular_v = min(angular_v, 0.6)
            angular_v = max(angular_v, -0.6)

    return linear_v,angular_v


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

    #pos, orient = env.robot.get_pos_orient(env.robot.left_end_effector)


    pos, orient = env.human.get_pos_orient(env.human.head)
    #print(pos)
    #print(orient)
    human_target = np.array(env.target_pos)
    print('target pose',human_target)

    mat = p.getMatrixFromQuaternion(orient)
    dir0 = [mat[0], mat[3], mat[6]]
    dir1 = [mat[1], mat[4], mat[7]]
    dir2 = [mat[2], mat[5], mat[8]]
        
    lineLen=1.0
    # works only for head  1.5 linlen
    dir2_neg = [-mat[1], -mat[4], -mat[7]]
    to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
        
    toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
    toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
    toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
        
    final_target_point = [ 0.5*(toX[0] + toY[1]), 0.5*(toX[1] + toY[1]), 0.5*(toX[2] + toY[2]) ]

    #-0.36859864 -0.06440637  0.80693936  1.
    #-0.36859864 -0.06440637  0.80693936  2.

    #human_target[0]=-0.3859
    #human_target[1]=0.46769
    #human_target[2]=0.849474
    print('-----------------------------------------')
    # Arrow keys for moving the base, 
    # s/x for the lift, third array
    # z/c for the prismatic joint, fourth array (compression/retraction)
    # a/d for the wrist joint, fifth array  (yaw movements)

    keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

    target_pose = env.target_pos

    target_pose_angles = env.robot.ik(env.robot.left_end_effector, final_target_point, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)

    print('robot target angle',target_pose_angles)

    current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

    print('current_joint_angles of robot',current_joint_angles)


    action = np.zeros(env.action_robot_len)
    right_side=False
    count_= 0

    setpoint_position = [-0.37606917, -1.00343241, 135]
    
    joint_anle_sp = [0.78297793, 0.07654554, 0.6533075]
    #yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
                                                                                                                                        #xyzw

    reward_total = 0
    while count_<250:
        env.render()
        action = np.zeros(env.action_robot_len)
 
        base_pos, base_orient = env.robot.get_pos_orient(env.robot.base)

        current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

        yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))

        base_pos[2] = yaw_orientation/math.pi*180

        #print('robot angle',yaw_orientation/math.pi*180)
        #print('robot target angle',setpoint_position)
        print('current_joint_angles of robot',current_joint_angles)

        linear_v, angular_v = position_controller(setpoint_position, base_pos)
        

        #print(env.robot.get_joint_angles(env.robot.left_arm_joint_indices))
        
        action = np.zeros(env.action_robot_len)
        action = action + linear_v*np.array([1, 1, 0, 0, 0])
        action = action + angular_v*np.array([1, -1, 0, 0, 0])

        #print('robot ', linear_v, angular_v, action)
        kp_angles = 2
        action[2:5] = (joint_anle_sp-current_joint_angles)*kp_angles

        # if count_>100:
        #     keys = p.getKeyboardEvents()
        #     for key, a in keys_actions.items():
        #         if key in keys and keys[key] & p.KEY_IS_DOWN:
        #             action += a
        #             action = action*100

        observation, reward, done, info = env.step(action)
        reward_total += reward
        task_success = info['task_success']

        count_ = count_+1

    print('reward')
    print(reward_total)
    print('task_success')
    print(task_success)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='ObjectHandoverStretch-v1',
                    help='Environment to test (default: JointReachingStretchHuman-v1)')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--seed', type=int, default=5,
                        help='Random seed (default: 5)')
    args = parser.parse_args()
    run_baseline()