import numpy as np
import pybullet as p
import optparse
from scipy.spatial.transform import Rotation as R
import os
import cv2

from assistive_gym.envs.agents.furniture import Furniture

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture
from .util import reward_base_direction
from .util import reward_tool_direction
#from .util import generate_line
#from .util import generate_line_hand

class JointMotionEnv(AssistiveEnv):

    def __init__(self, robot, human):
        super(JointMotionEnv, self).__init__(robot=robot, human=human, task='joint_motion', obs_robot_len=(23 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(24 + len(human.controllable_joint_indices)))
        


    def step(self, action):

        self.distance_threshold = 0.1/1.5

        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
            print(action)
        
        #action[4]=500
        #print('Full in JM', action)
        self.take_step(action*100)

        obs = self._get_obs()
        # print(np.array_str(obs, precision=3, suppress_small=True))

        # print(action)
        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_at_target)

        tool_pos = self.tool.get_base_pos_orient()[0]
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos) #Penalize distances away from target
        reward_action = -np.linalg.norm(action) #Penalize actions
        reward_force_scratch = 0.0 #Reward force near the target

        #if self.target_contact_pos is not None and np.linalg.norm(self.target_contact_pos - self.prev_target_contact_pos) > 0.001 and self.tool_force_at_target < 10:
        #Encourage the robot to move around near the target to simulate scratching

        #Discourage movement only for stretch not for pr2
        self.robot_current_pose,robo_orient_ = self.robot.get_pos_orient(self.robot.base)

        reward_movement = -1*np.linalg.norm(self.robot_current_pose-self.robot_old_pose)
        
        #only for pr2 not stretch
        #reward_movement=0


        self.robot_old_pose = self.robot_current_pose

        self.robot_current_arm,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        
        reward_shake = 0
        if self.task_success>1:
            reward_shake = -1*np.linalg.norm(self.robot_current_arm-self.robot_old_arm)
        
        self.robot_old_arm = self.robot_current_arm


        head_pose,head_orient = self.human.get_pos_orient(self.human.head)
        wrist_pos,wrist_orient = self.human.get_pos_orient(self.human.right_wrist)
        robot_tool_pos, tool_orient = self.tool.get_base_pos_orient()


        if reward_base_direction(head_pose, head_orient, self.robot_current_pose):
            reward_robot_orientation = 5
            print('Robot in orientation with human')
        else:
            reward_robot_orientation = 0


        if reward_tool_direction(wrist_pos, wrist_orient , robot_tool_pos):
            reward_tool_orientation = 3
        else:
            reward_tool_orientation = 0
        

        #print(self.robot_current_pose)
        if abs(reward_distance)<self.distance_threshold: #0.03
            reward_force_scratch = 5
            self.task_success = 1

        if abs(reward_distance)<self.distance_threshold and self.tool_force_at_target<10: #0.03
            reward_force_scratch = 5
            self.task_success =+ 2

        # if self.total_force_on_human>10:
        #     print('Force on human: ',self.total_force_on_human)

        ######### Generate line 
        p.removeAllUserDebugItems()
        head_t_pos,head_t_orient = self.human.get_pos_orient(self.human.head)
        #self.generate_line(head_t_pos,head_t_orient)
        wrt_pos,wrt_orient = self.human.get_pos_orient(self.human.right_wrist)
        #self.generate_line_hand(wrt_pos,wrt_orient,0.3)

        tool_pos, tool_orient = self.tool.get_base_pos_orient()
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient) #useless command
        #self.generate_line(self.robot_current_pose,robo_orient_)
        #self.generate_line(tool_pos,tool_orient,0.3)


        ##########
        reward = 0
        reward = reward_shake + reward_movement +  self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('scratch_reward_weight')*reward_force_scratch + preferences_score
        reward = reward + self.config('robot_orientation')*reward_robot_orientation + self.config('tool_orientation')*reward_tool_orientation

        self.total_reward=self.total_reward+reward
        self.prev_target_contact_pos = self.target_contact_pos

        if self.gui and self.tool_force_at_target > 0:
            print('Task success:', self.task_success, 'Tool force at target:', self.tool_force_at_target, reward_force_scratch)
            print('Task reward:', self.total_reward)

        #info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = {'total_force_on_human': self.total_force_on_human, 'task_success': self.task_success, 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        jt = self.robot.get_joint_angles(indices=self.robot.left_arm_joint_indices)
        #print('robot joint angles',jt)

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}




    def generate_line(self, pos, orient, lineLen=1.5):
        
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
        to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
        to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
        
        toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
        toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
        toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
        
        p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
        p.addUserDebugLine(pos, toY, [0, 1, 0], 5)
        p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)

        p.addUserDebugLine(pos, to1, [0, 1, 1], 5, 3)
        p.addUserDebugLine(pos, to2, [0, 1, 1], 5, 3)
        p.addUserDebugLine(to2, to1, [0, 1, 1], 5, 3)



    def generate_line_hand(self, pos, orient, lineLen=1.5):
        
        mat = p.getMatrixFromQuaternion(orient)
        dir0 = [mat[0], mat[3], mat[6]]
        dir1 = [mat[1], mat[4], mat[7]]
        dir2 = [mat[2], mat[5], mat[8]]
        
        # works only for hand 0.25 linelen
        dir2_neg = [-mat[2], -mat[5], -mat[8]]
        to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
        to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
        
        # works only for head  1.5 linlen
        # dir2_neg = [-mat[1], -mat[4], -mat[7]]
        # to1 = [pos[0] + lineLen * (dir2_neg[0]+dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]+dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
        # to2 = [pos[0] + lineLen * (dir2_neg[0]-dir0[0])/2, pos[1] + lineLen * (dir2_neg[1]-dir0[1])/2, pos[2] + lineLen * (dir2_neg[2]+dir0[2])/2 ]
        
        toX = [pos[0] + lineLen * dir0[0], pos[1] + lineLen * dir0[1], pos[2] + lineLen * dir0[2]]
        toY = [pos[0] + lineLen * dir1[0], pos[1] + lineLen * dir1[1], pos[2] + lineLen * dir1[2]]
        toZ = [pos[0] + lineLen * dir2[0], pos[1] + lineLen * dir2[1], pos[2] + lineLen * dir2[2]]
        
        p.addUserDebugLine(pos, toX, [1, 0, 0], 5)
        p.addUserDebugLine(pos, toY, [0, 1, 0], 5)
        p.addUserDebugLine(pos, toZ, [0, 0, 1], 5)

        p.addUserDebugLine(pos, to1, [0, 1, 1], 5, 3)
        p.addUserDebugLine(pos, to2, [0, 1, 1], 5, 3)
        p.addUserDebugLine(to2, to1, [0, 1, 1], 5, 3)



    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_at_target = 0
        target_contact_pos = None
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            # Enforce that contact is close to the target location
            if linkA in [0, 1] and np.linalg.norm(posB - self.target_pos) < self.distance_threshold:
                tool_force_at_target += force
                target_contact_pos = posB
        return total_force_on_human, tool_force, tool_force_at_target, None if target_contact_pos is None else np.array(target_contact_pos)



    def _get_obs(self, agent=None):
        tool_pos, tool_orient = self.tool.get_base_pos_orient()
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]#5upperarm   agym_jt[6,:] 
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]#7forearm          agym_jt[8,:]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]#9j_right_wrist_x  agym_jt[10,:]  

        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)

        #human_pose_predicted = self.call_bp_cb()
        #print("Human pose predicted", shoulder_pos- human_pose_predicted[6] )

        self.total_force_on_human, self.tool_force, self.tool_force_at_target, self.target_contact_pos = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, tool_pos_real - target_pos_real, target_pos_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            print("human:", human_joint_angles)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, tool_pos_human - target_pos_human, target_pos_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.total_force_on_human, self.tool_force_at_target]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs




    def reset(self):
        super(JointMotionEnv, self).reset()

        self.build_assistive_env('wheelchair')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            #self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        # Update robot and human motor gains
        self.robot.motor_gains = self.human.motor_gains = 0.005

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, -self.np_random.uniform(0, 10)), (self.human.j_head_y, -self.np_random.uniform(0, 10)), (self.human.j_head_z, -self.np_random.uniform(0, 10))]
        
        # j_right_pecs_x, j_right_pecs_y, j_right_pecs_z, j_right_shoulder_x, j_right_shoulder_y, j_right_shoulder_z, j_right_elbow, j_right_forearm, j_right_wrist_x, j_right_wrist_y
        # human_start = [-1.43457602e-02,  4.88734760e-04,  5.31665415e-01,  1.34692067e-01, -9.64689935e-02, -2.42673015e-01, -1.64185327e+00,  2.51781710e-02,  6.91561229e-03,  1.17776584e-01]
        right_joint_list = [self.human.j_right_pecs_x, self.human.j_right_pecs_y, self.human.j_right_pecs_z,self.human.j_right_shoulder_x,self.human.j_right_shoulder_y, self.human.j_right_shoulder_z, self.human.j_right_elbow, self.human.j_right_forearm, self.human.j_right_wrist_x, self.human.j_right_wrist_y]
        
        # start
        right_arm_angles = [0.0,  0.0,  5.31665415e-01,  1.34692067e-01, -9.64689935e-02, -2.42673015e-01, -1.64185327e+00,  0.0,  0.0,  0.0]
        
        #end
        # right_arm_angles = [0.0,  0.0, 0.0,  0.6285238,  -1.2184946,   0.0,  -2.2070327,  0.0, 0.0, 0.0]

        right_arm_pos = [(right_joint_list[i], right_arm_angles[i]*180/np.pi) for i in range(len(right_joint_list))]
        joints_positions += right_arm_pos
       
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        chest_pos, chest_orient = self.human.get_pos_orient(self.human.stomach)
        ctarget_pos, ctarget_orient = p.multiplyTransforms(chest_pos, chest_orient, [0,0,0], [0, 0, 0, 1], physicsClientId=self.id)

        #self.create_sphere(radius=0.4, mass=0.0, pos=ctarget_pos, visual=True, collision=False, rgba=[1, 0, 0, 0.3]

        self.table = Furniture()
        #self.chair.init(7, env.id, env.np_random, indices=-1)
        self.table.init(furniture_type='table', directory=self.directory, id=self.id, np_random=self.np_random)
        self.table.set_base_pos_orient([0.3, -0.85, 0.0], [0, 0, 0])
        self.table.set_gravity(0, 0, -9.8)

        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=55, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.045]*3, alpha=0.75)
        self.robot.skip_pose_optimization = True
        target_ee_pos = np.array([-0.1, 0.1, 0.5]) 
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        #self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human, self.furniture])

        pos = [-0.625, -0.5, 0.1]
        # pos = [-1.825, -0.5, 0.1]
        orient = [0, 0, np.pi / 2.0]
        self.robot.set_base_pos_orient(pos, orient)
        self.robot.randomize_init_joint_angles(self.task)

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        self.bowl = Furniture()
        self.bowl.init('bowl', self.directory, self.id, self.np_random)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, -9)

        # Generate food
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        bowl_pos, bowl_orient = self.bowl.get_pos_orient(self.bowl.base)
        #bowl_pos = [-0.1, 0, 0]
        food_radius = 0.005
        food_mass = 0.001
        batch_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    batch_positions.append(np.array([i*2*food_radius, j*2*food_radius, k*2*food_radius+0.01]) + bowl_pos)
        self.foods = self.create_spheres(radius=food_radius, mass=food_mass, batch_positions=batch_positions, visual=False, collision=True)
        colors = [[60./256., 186./256., 84./256., 1], [244./256., 194./256., 13./256., 1],
                  [219./256., 50./256., 54./256., 1], [72./256., 133./256., 237./256., 1]]
        for i, f in enumerate(self.foods):
            p.changeVisualShape(f.body, -1, rgbaColor=colors[i%len(colors)], physicsClientId=self.id)
        self.total_food_count = len(self.foods)
        self.foods_active = [f for f in self.foods]

        # Enable rendering
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop food in the spoon
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        #print('Robot is mobile or not', self.robot.mobile)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()



    
    def generate_target(self):

        self.total_reward=0
        
        self.robot_old_pose,_orient_ = self.robot.get_pos_orient(self.robot.base)
        self.robot_old_arm,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)

        if self.human.gender == 'male':
            self.limb, length, radius = self.human.right_elbow, 0.157, 0.033
        else:
            self.limb, length, radius = self.human.right_elbow, 0.134, 0.027
           
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, -2*length]), p2=np.array([0, 0, -length*2.5]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
        print('Target pose 1', target_pos)
        target_pos_2 = [+0.115, -0.45, 0.73]
        self.create_sphere(radius=0.02, mass=0.0, pos=target_pos_2, visual=True, collision=False, rgba=[1, 0, 1, 1])
        
        target_pos_3 = [-0.07, -0.41, 0.73]
        self.create_sphere(radius=0.02, mass=0.0, pos=target_pos_3, visual=True, collision=False, rgba=[0, 1, 1, 1])
        #self.create_sphere(radius=0.1, mass=0.0, pos=arm_pos, visual=True, collision=False, rgba=[0, 0, 1, 1])

        self.update_targets()



    def update_targets(self):
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target_orient = np.array(target_orient)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])
        #print('Handover pose in main environment ',self.target_pos)

    
    def get_depth_image(self):
        # * take image after reward computed
        img, depth = self.get_camera_image_depth()

        far = 2.406
        near = 1.406

        a_ = (far-near)/(np.max(depth) - np.min(depth))
        b_ = - (far*np.min(depth)-near*np.max(depth))/(np.max(depth) - np.min(depth))
        depth = depth*(a_)+b_

        depth = depth*1000

        depth = depth[50:178, 73:127]

        #filename='/nethome/nnagarathinam6/Documents/joint_reaching_evaluation/'
        #outfile = filename + "after_depth" + str(1) + ".npy"
        #np.save(outfile, depth)
        
        #file = filename + "depth" + str(1) + ".png"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('color', img)
        cv2.waitKey(0) 

        return depth

