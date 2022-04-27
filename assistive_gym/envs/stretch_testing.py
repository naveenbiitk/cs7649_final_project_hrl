import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture
from .agents.stretch import Stretch
from math import sin,cos

robot_arm = 'left'

# class ScratchItchStretchEnv(ScratchItchEnv):
#     def __init__(self):
#         super(ScratchItchStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))


class StretchTestingEnv(AssistiveEnv):

    def __init__(self):
        super(StretchTestingEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=None, task='stretch_testing', obs_robot_len=11, obs_human_len=0)
        self.robot_obs_list = []

    def step(self, action):
 
        self.take_step(action)
        obs = self._get_obs()
       
        self.robot_current_pose,base_orient = self.robot.get_pos_orient(self.robot.base)
        tool_pos,_ = self.robot.get_pos_orient(self.robot.left_tool_joint) 


        # reward_shake = -np.linalg.norm(self.old_tool_pos - tool_pos)
        # reward_base_movement = -np.linalg.norm(self.robot_current_pose-self.robot_old_pose)
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos) #Penalize distances away from target
        # reward_base_distance = -np.linalg.norm(self.target_pos[:2] - self.robot_current_pose[:2])
        # reward_action = -np.linalg.norm(action) #Penalize actions

        base_pos = self.robot_current_pose
        pf = self.target_pos
        yaw_orientation = np.arctan2(2*(base_orient[3]*base_orient[2]+base_orient[0]*base_orient[1]),1-2*(base_orient[1]*base_orient[1]+base_orient[2]*base_orient[2]))
        base_pos_set = [pf[0]-0.0*cos(yaw_orientation)-0.4,pf[1]-0.0*sin(yaw_orientation)-0.1,base_pos[2]]
 
        reward_base_distance = -np.linalg.norm(base_pos_set - self.robot_current_pose)


        current_joint_angles = self.robot.get_joint_angles(self.robot.left_arm_joint_indices)
        target_joint_angles = self.robot.ik(self.robot.left_tool_joint, pf, base_orient, ik_indices=self.robot.left_arm_ik_indices, max_iterations=200)
        
        reward_target_angles = -np.linalg.norm(target_joint_angles - current_joint_angles)

        reward_time = -self.iteration



        #print('Tool pose', tool_pos, ' Difference',reward_distance)
        if abs(reward_distance) < 0.5:
            self.task_success = 1
            print('Task success done!')
        else:
            self.task_success = 0

        #reward = self.config('base_distance_weight')*reward_base_distance + self.config('task_success_threshold')*self.task_success + self.config('shake_weight')*reward_shake + self.config('base_move_weight')*reward_base_movement +  self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action 
        reward = self.config('base_distance_weight')*reward_base_distance + self.config('task_success_threshold')*self.task_success  + self.config('tangle_weight')*reward_target_angles +  self.config('time_weight')*reward_time 


        self.total_reward = self.total_reward + reward
 
        if self.gui:
            print('Task success:', self.task_success, 'Task total reward:', self.total_reward, ' Static reward', reward )

        #info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = { 'task_success': self.task_success, 'action_robot_len': self.action_robot_len,  'obs_robot_len': self.obs_robot_len}
        

        self.robot_old_pose = self.robot_current_pose
        self.old_tool_pos = tool_pos
        #print('Observation space ', np.array(self.robot_obs_list).shape)
        if self.iteration>=200 or self.task_success==1:
            done = True
            #np.save('robot_observation_space.npy', np.array(self.robot_obs_list) )
        else:
            done = False

        return obs, reward, done, info

   

    def _get_obs(self, agent=None):

        target_pos_real_static = self.target_pos
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)

        

        base_pos, base_orient_quat = self.robot.get_base_pos_orient()
        
        base_orient_euler = np.array(p.getEulerFromQuaternion(np.array(base_orient_quat), physicsClientId=self.id))

        pose_orient = np.array([base_pos[0],base_pos[1],base_orient_euler[2]])
        #print('Robot euler angles',int(base_orient_euler[2]/np.pi*180) )

        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi

        robot_obs = np.concatenate([target_pos_real_static,pose_orient, robot_joint_angles]).ravel()


        #print('Robot observations ', robot_obs)
        self.robot_obs_list.append(robot_obs)
        return robot_obs

    def reset(self):
        super(StretchTestingEnv, self).reset()
        self.build_assistive_env()


        target_ee_pos = np.array([1.25, 0.25, 0]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion([0, 0, 2*np.random.random()])
        #target_ee_orient = np.array([0, 0, 0, 1 ])
        # Not in itch scratch only here
        #self.robot.reset_joints()
        self.robot.skip_pose_optimization = True
        #self.init_robot_pose(target_ee_pos, target_ee_orient, [None], [None], arm='left', tools=[], collision_objects=[],wheelchair_enabled=True)
        
        self.robot.randomize_init_joint_angles(self.task)
        self.robot.set_base_pos_orient(target_ee_pos, target_ee_orient)

        #self.init_robot_pose(target_ee_pos, target_ee_orient, start_pos_orient, target_pos_orients, arm='right', tools=[], collision_objects=[], wheelchair_enabled=True, right_side=True, max_iterations=3)
        self.generate_target()

        #p.setGravity(0, 0, -9.81, physicsClientId=self.id)  #changes the whole simulation response
        #if not self.robot.mobile:
        self.robot.set_gravity(0, 0, -9.81)

        print('Robot pose',self.robot.get_pos_orient(self.robot.base) , 'Target pose', self.target_pos)
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    
    def generate_target(self):

        self.total_reward = 0
        self.robot_old_pose,_ = self.robot.get_pos_orient(self.robot.base)
        self.old_tool_pos,_ = self.robot.get_pos_orient(self.robot.left_end_effector)


        target_pos = [ 0.5+np.random.random(), 0.5+np.random.random(), 0.7+(np.random.random()/10)]
        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

        #target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        #self.create_sphere(radius=0.1, mass=0.0, pos=arm_pos, visual=True, collision=False, rgba=[0, 0, 1, 1])




