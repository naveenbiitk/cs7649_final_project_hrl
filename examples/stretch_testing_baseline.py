import gym, assistive_gym, argparse
import pybullet as p
import numpy as np
import sys
import math
import twodcontroller as tc
from math import sin,cos

#angular pid constansts
prev_angle_term = 0
integral_object_angle = 0
#linear pid constants
prev_linear_term = 0
integral_linear_gap = 0


def linear_controller(object_gap):
    kp_gap = 4.25/2.3
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
    kp_angle_gap = 4.02/4.5
    kd_angle_gap = 0.05#5
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
            #d_del = np.max(np.abs(np.array(setpoint_position[:2])-np.array(current_position[:2])))
            d_del = np.linalg.norm( np.array(setpoint_position[:2])-np.array(current_position[:2]) )

            linear_v = linear_controller(d_del)
            angular_v = angular_controller(theta_del)
            #print(' Position change: ', error_position, linear_v)
        elif min(abs(error_theta),360-abs(error_theta)) > theta_threshold:
            #print('Angular control')
            angular_v = angular_controller(np.array(setpoint_position[2])-np.array(current_position[2]))
        
        angular_v = min(angular_v, 0.8)
        angular_v = max(angular_v, -0.8)

    k_array = np.array([ [  0.8,  0.8 ,0.08],
                         [0.01,0.01, 0.1] ])
    error_array = np.array(setpoint_position)-np.array(current_position)

    final_array = np.matmul(k_array, error_array)
    linear_v = final_array[0]
    angular_v = final_array[1]
    print('Final array', final_array[0])

    return linear_v,angular_v



EI_term=0
old_e=0
Kp=4
Ki=0
Kd=0.02
v_constant=0.8




def iteratePID(diff,cur_theta):
    global EI_term,old_e,Kp,Ki,Kd,v_constant

    dx = diff[0]
    dy = diff[1]

    g_theta = np.arctan2(dy,dx)

    alpha = g_theta-cur_theta
    e = np.arctan2(np.sin(alpha), np.cos(alpha))

    e_P = e
    e_I = EI_term + e
    e_D = e - old_e
    w = Kp*e_P + Ki*e_I + Kd*e_D

    w = np.arctan2(np.sin(w), np.cos(w))

    EI_term = EI_term + e
    old_e = e
    v = v_constant

    return v,w



def runPID(setpoint_position, current_position):

    arrive_distance=0.05

    x = current_position[0]
    y = current_position[1]
    theta = current_position[2]
    diff = np.array(setpoint_position[:2])-np.array(current_position[:2])
    diff_err = diff@diff.T
    v = 0
    w = 0
    #print('diff_err',diff_err)
    if(diff_err<arrive_distance):
        v, w = iteratePID(diff,theta)

    return v,w



def manipulator_control(sp_end, end_efffector):
    kp_lift = 1
    kp_ext = 1
    kp_yaw = 1

    lift_sp = sp_end[2]
    lift_curret = end_efffector[2]

    #ext_sp = sp_end[1]
    #ext_curret = end_efffector[1]
    ext_err = np.linalg.norm(np.array(sp_end[:2])-np.array(end_efffector[:2]))

    yaw_sp = sp_end[2]
    yaw_curret = end_efffector[2]

    lift_action = kp_lift*(lift_sp - lift_curret)
    ext_action = kp_ext*(ext_err)
    yaw_action = kp_yaw*(yaw_sp - yaw_curret)*0

    return [lift_action, ext_action, yaw_action]


def manipulator_control(sp_end, end_efffector):
    target_joint_angles = self.ik(self.left_end_effector, target_pos, target_orient, ik_indices=self.left_arm_ik_indices, max_iterations=max_iterations, half_range=self.half_range, randomize_limits=(randomize_limits and r >= 10))    
    self.set_joint_angles(self.left_arm_joint_indices, target_joint_angles)



def generate_target_point(pos, orient):
    lineLen=0.5
    mat = p.getMatrixFromQuaternion(orient)
    dir0 = [mat[0], mat[3], mat[6]]
    dir2_neg = [-mat[1], -mat[4], -mat[7]]
    k1 = np.random.random()+0.3
    k2 = np.random.random()+0.3
    k3 = np.random.random()+0.3
    p1 = [pos[0] + k1*lineLen * (0.9*dir2_neg[0]+0.1*dir0[0]), pos[1] + k1*lineLen * (0.9*dir2_neg[1]+0.1*dir0[1]), pos[2] + k1*lineLen * (0.9*dir2_neg[2]+0.1*dir0[2])/2 ]
    p2 = [pos[0] + k2*lineLen * (0.9*dir2_neg[0]-0.1*dir0[0]), pos[1] + k2*lineLen * (0.9*dir2_neg[1]-0.1*dir0[1]), pos[2] + k2*lineLen * (0.9*dir2_neg[2]+0.1*dir0[2])/2 ]
    pf = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2,(p1[2]+p2[2])/2]
    pf[2] = k3
    
    return pf


def generate_line(pos, orient, lineLen=1.0):
        
    mat = p.getMatrixFromQuaternion(orient)
    dir0 = [mat[0], mat[3], mat[6]]
    dir1 = [mat[1], mat[4], mat[7]]
    dir2 = [mat[2], mat[5], mat[8]]
    
    # works only for hand 0.25 linelen
    #dir2_neg = [-mat[2], -mat[5], -mat[8]]
    #to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    #to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
    
    # works only for head  1.5 linlen
    dir2_neg = [-mat[1], -mat[4], -mat[7]]

    to1 = [pos[0] + lineLen * (0.9*dir2_neg[0]+0.1*dir0[0]), pos[1] + lineLen * (0.9*dir2_neg[1]+0.1*dir0[1]), pos[2] + lineLen * (0.9*dir2_neg[2]+0.1*dir0[2])/2 ]
    to2 = [pos[0] + lineLen * (0.9*dir2_neg[0]-0.1*dir0[0]), pos[1] + lineLen * (0.9*dir2_neg[1]-0.1*dir0[1]), pos[2] + lineLen * (0.9*dir2_neg[2]+0.1*dir0[2])/2 ]
    
    toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
    toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
    toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
        

    #p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
    #p.addUserDebugLine(pos, toY, [0, 1, 0], 5)
    #p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)
    p.addUserDebugLine(pos, to1, [0, 1, 1], 5, 3)
    p.addUserDebugLine(pos, to2, [0, 1, 1], 5, 3)
    p.addUserDebugLine(to2, to1, [0, 1, 1], 5, 3)


def run_baseline():
    print('--------------------------------------')
    env = gym.make(args.env)
    seed=2
    env.seed(seed)
    env.render()
    observation = env.reset()
    #env.robot.print_joint_info()

    print('--------------------------------------')
    print('robot base pose',env.robot.get_pos_orient(env.robot.base))


    final_target = np.array(env.target_pos)
    print('target pose',final_target)
        
    #final_target_point = [ 0.5*(toX[0] + toY[1]), 0.5*(toX[1] + toY[1]), 0.5*(toX[2] + toY[2]) ]

    print('-----------------------------------------')

    keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

    target_pose = env.target_pos+[0,0,0.1]
    target_orient = [0, 0, 0, 1]

    target_pose_angles = env.robot.ik(env.robot.left_end_effector, target_pose, target_orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)

    print('robot target angle',target_pose_angles)

    current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

    print('current_joint_angles of robot',current_joint_angles)

    action = np.zeros(env.action_robot_len)
    right_side=False
    count_= 0


    joint_angle_sp = target_pose_angles - [0, -0.2, 1]
    #yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
                                                                                                                                        #xyzw
    init_pos, init_orient = env.robot.get_pos_orient(env.robot.base)
    #print('Initial values ',init_pos,' other ',target_pose+[0,0,0.3], target_pose-init_pos)
    print('Expected: ',[0.68297793, 0.07654554, 0.6533075],'Calculated', joint_angle_sp)

    reward_total = 0

    #init robot base pose (array([0.26500225, 0.26230312, 0.01825219]), array([0.        , 0.        , 0.47724787, 0.87876873]))
    #target pose [0.4299674  0.9219356  0.70425493]
    #Final base position [ 0.51615957  0.79551299 99.2948529 ]

    #setpoint_position = [final_target[0], final_target[1]-0.1, 100]
    setpoint_position = [0.4299674,  0.9219356-0.1, 100]
    

    base_pos, base_orient = env.robot.get_pos_orient(env.robot.base)
    pf = generate_target_point(base_pos,base_orient)
    pf = env.target_pos
    yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
    base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-0.4,pf[1]-0.0*sin(yaw_orientation)-0.1,base_pos[2]]
    
    env.robot.set_base_pos_orient( base_pos_set, base_orient)
    


    while count_<200:
        env.render()

        action = np.zeros(env.action_robot_len)
 
        base_pos, base_orient = env.robot.get_pos_orient(env.robot.base)

        #generate_target_point(pos, orient)
        #current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

        yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))

        base_pos[2] = yaw_orientation/math.pi*180

        #print('robot angle',yaw_orientation/math.pi*180)
        #print('robot target angle',setpoint_position)
        #print('Current_joint_angles of robot',current_joint_angles)
        #print('Current base position',base_pos)
        #linear_v, angular_v = position_controller(setpoint_position, base_pos)
        #linear_v,angular_v = runPID(setpoint_position, base_pos)
        #wrist_pos, wrist_orient = env.robot.get_pos_orient(env.robot.left_gripper_indices)

        #action_part = manipulator_control(final_target,wrist_pos)
        #print(env.robot.get_joint_angles(env.robot.left_arm_joint_indices))
        
        action = np.zeros(env.action_robot_len)
        
        
        
        
        #env.robot.ik(env.robot.left_end_effector, final_target_point, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)left_gripper_collision_indices

        current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

        target_joint_angles = env.robot.ik(env.robot.left_tool_joint, pf, base_orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)
        env.robot.set_joint_angles(env.robot.left_arm_joint_indices, target_joint_angles)
        


        #print('target_joint_angles ', target_joint_angles)
        #action = action + linear_v*np.array([1, 1, 0, 0, 0])
        #action = action + angular_v*np.array([1, -1, 0, 0, 0])

        #print('robot ', linear_v, angular_v, action)
        #kp_angles = 2
#        action[2:5] = action_part
        #action = np.zeros(env.action_robot_len)
        # if count_>300:
        #     keys = p.getKeyboardEvents()
        #     for key, a in keys_actions.items():
        #         if key in keys and keys[key] & p.KEY_IS_DOWN:
        #             action_1 += a
        #             action_1 = action_1*10

        wrist_pos, wrist_orient = env.robot.get_pos_orient(env.robot.left_tool_joint)
        print('Wr',wrist_pos,'target',pf,'diff',wrist_pos-pf)
        env.create_sphere(radius=0.02, mass=0.0, pos=wrist_pos, visual=True, collision=False, rgba=[1, 1, 1, 1])

        #action = action
        observation, reward, done, info = env.step(action)

        reward_total += reward
        task_success = info['task_success']

        count_ = count_+1
        if done:
            break
    print('reward')
    print(reward_total)
    print('task_success')
    print(task_success)


def evaluate_baseline():
    seed=0

    n_episodes = 2
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
        pos, orient = env.human.get_pos_orient(env.human.head)
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

        #print('-----------------------------------------')

        keys_actions = {p.B3G_LEFT_ARROW: np.array([0.01, -0.01, 0, 0, 0]), p.B3G_RIGHT_ARROW: np.array([-0.01, 0.01, 0, 0, 0]), p.B3G_UP_ARROW: np.array([0.01, 0.01, 0, 0, 0]), p.B3G_DOWN_ARROW: np.array([-0.01, -0.01, 0, 0, 0]), ord('s'): np.array([0, 0, 0.01, 0, 0]), ord('x'): np.array([0, 0, -0.01, 0, 0]), ord('z'): np.array([0, 0, 0, -0.01, 0]), ord('c'): np.array([0, 0, 0, 0.01, 0]), ord('a'): np.array([0, 0, 0, 0, 0.01]), ord('d'): np.array([0, 0, 0, 0, -0.01])}

        target_pose = env.target_pos

        target_pose_angles = env.robot.ik(env.robot.left_end_effector, final_target_point, orient, ik_indices=env.robot.left_arm_ik_indices, max_iterations=200)

        #print('robot target angle',target_pose_angles)

        current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)

        #print('current_joint_angles of robot',current_joint_angles)

        action = np.zeros(env.action_robot_len)
        right_side=False
        count_= 0

        #setpoint_position = [-0.37606917, -1.00343241, 135]
        setpoint_position = [-0.37606917, -0.920343241, 145]
        joint_anle_sp = [0.68297793, 0.07654554, 0.6533075]
        #yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
                                                                                                                                        #xyzw
        init_pos, init_orient = env.robot.get_pos_orient(env.robot.base)
        #print('Initial values ',init_pos,' other ',target_pose, target_pose-init_pos)

#---------------------------------------------------------------------------------------------------------------------------------------
        reward_total = 0
        while count_<200:
            env.render()
            action = np.zeros(env.action_robot_len)
 
            base_pos, base_orient = env.robot.get_pos_orient(env.robot.base)
    
            current_joint_angles = env.robot.get_joint_angles(env.robot.left_arm_joint_indices)
    
            yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
    
            base_pos[2] = yaw_orientation/math.pi*180
    
            #print('robot angle',yaw_orientation/math.pi*180)
            #print('robot target angle',setpoint_position)
            #print('current_joint_angles of robot',current_joint_angles)

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
    
            action = action*2
            observation, reward, done, info = env.step(action)
            reward_total += reward
            task_success = info['task_success']

            count_ = count_+1
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env', default='StretchTesting-v1',
                    help='Environment to test (default: JointReachingStretchHuman-v1)')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--seed', type=int, default=5,
                        help='Random seed (default: 5)')
    args = parser.parse_args()
    
    run_baseline()
    #evaluate_baseline()