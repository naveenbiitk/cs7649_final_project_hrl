import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture


class JointReachingEnv(AssistiveEnv):

    def __init__(self, robot, human):
        super(JointReachingEnv, self).__init__(robot=robot, human=human, task='joint_reaching', obs_robot_len=(23 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(24 + len(human.controllable_joint_indices)))
        self.robot_obs_list = []

    def step(self, action):

        self.distance_threshold = 0.1/1.5

        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()
        # print(np.array_str(obs, precision=3, suppress_small=True))

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_at_target)

        tool_pos = self.tool.get_pos_orient(1)[0]
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos) # Penalize distances away from target
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_force_scratch = 0.0 # Reward force near the target
        #if self.target_contact_pos is not None and np.linalg.norm(self.target_contact_pos - self.prev_target_contact_pos) > 0.001 and self.tool_force_at_target < 10:
        # Encourage the robot to move around near the target to simulate scratching
        

        # Discourage movement only for stretch not for pr2
        self.robot_current_pose,_orient_ = self.robot.get_pos_orient(self.robot.base)
        reward_movement = -3*np.linalg.norm(self.robot_current_pose-self.robot_old_pose)
        
        #only for pr2 not stretch
        #reward_movement=0

        self.robot_old_pose = self.robot_current_pose

        self.robot_current_arm,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        
        reward_shake = 0
        if self.task_success>1:
            reward_shake = -1*np.linalg.norm(self.robot_current_arm-self.robot_old_arm)
        
        self.robot_old_arm = self.robot_current_arm
            

        #print(self.robot_current_pose)
        if abs(reward_distance)<self.distance_threshold: #0.03
            reward_force_scratch = 5
            self.task_success = 1

        if abs(reward_distance)<self.distance_threshold and self.tool_force_at_target<10: #0.03
            reward_force_scratch = 5
            self.task_success =+ 2

        # if self.total_force_on_human>10:
        #     print('Force on human: ',self.total_force_on_human)


        reward = reward_shake + reward_movement +  self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('scratch_reward_weight')*reward_force_scratch + preferences_score

        self.total_reward=self.total_reward+reward
        self.prev_target_contact_pos = self.target_contact_pos

        if self.gui and self.tool_force_at_target > 0:
            print('Task success:', self.task_success, 'Tool force at target:', self.tool_force_at_target, reward_force_scratch)
            print('Task reward:', self.total_reward)

        #info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = {'total_force_on_human': self.total_force_on_human, 'task_success': self.task_success, 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        
        #print('Observation space ', np.array(self.robot_obs_list).shape)
        if self.iteration>=250:
            done = True
            #np.save('robot_observation_space.npy', np.array(self.robot_obs_list) )
        else:
            done = False

        jt = self.robot.get_joint_angles(indices=self.robot.left_arm_joint_indices)
        #print('robot joint angles',jt)

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

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
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if not self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        self.total_force_on_human, self.tool_force, self.tool_force_at_target, self.target_contact_pos = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, tool_pos_real - target_pos_real, target_pos_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        
        print('--------------------------------------------------')
        print('tool_pos_real ', np.array(tool_pos_real))
        print('tool_orient_real ', np.array(tool_orient_real ))
        print('difference ', np.array(tool_pos_real - target_pos_real ))
        print('target_pos_real ', np.array(target_pos_real ))
        print('robot_joint_angles ', np.array(robot_joint_angles ))
        print('shoulder_pos_real ', np.array(shoulder_pos_real ))
        print('Elbow_pos_real ', np.array(shoulder_pos_real ))
        print('Wrist_pos_real ', np.array(shoulder_pos_real ))
        print('Force ', np.array( [self.tool_force]))

        print('--------------------------------------------------')
        print('--------------------------------------------------')

        self.robot_obs_list.append(robot_obs)

        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
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
        super(JointReachingEnv, self).reset()
        self.build_assistive_env('bed', fixed_human_base=False)

        self.furniture.set_friction(self.furniture.base, friction=5)

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(self.human.j_right_shoulder_x, 30)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2.0, 0, 0])

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, self.np_random.uniform(-0.01, 0.01, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        self.prev_target_contact_pos = np.zeros(3)
        # self.robot.print_joint_info()

        # Lock human joints and set velocities to 0
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        # # Set joint angles for human joints (in degrees)
        # joints_positions = [(self.human.j_right_shoulder_x, 30), (self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        # self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None if self.human.controllable else 1, reactive_gain=0.01)

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[0.001]*3)

        target_ee_pos = np.array([-0.6, 0, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        #target_ee_orient = np.array([0, 0, 0, 1 ])
        # Not in itch scratch only here
        #self.robot.reset_joints()
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], arm='left', tools=[self.tool], collision_objects=[self.human, self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        self.generate_target()

        #p.setGravity(0, 0, -9.81, physicsClientId=self.id)  #changes the whole simulation response
        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, -1)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    # def generate_target(self):
    #     self.total_reward=0
        
    #     self.robot_old_pose,_orient_ = self.robot.get_pos_orient(self.robot.base)
    #     self.robot_old_arm,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)

    #     # Randomly select either upper arm or forearm for the target limb to scratch
    #     if self.human.gender == 'male':
    #         self.limb, length, radius = [[self.human.right_shoulder, 0.279, 0.043], [self.human.right_elbow, 0.257, 0.033]][self.np_random.randint(2)]
    #     else:
    #         self.limb, length, radius = [[self.human.right_shoulder, 0.264, 0.0355], [self.human.right_elbow, 0.234, 0.027]][self.np_random.randint(2)]
    #     self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, theta_range=(0, np.pi*2))
    #     arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
    #     target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

    #     self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

    #     self.update_targets()

    
    def generate_target(self):

        self.total_reward=0
        
        self.robot_old_pose,_orient_ = self.robot.get_pos_orient(self.robot.base)
        self.robot_old_arm,_orient_ = self.robot.get_pos_orient(self.robot.left_end_effector)
        # Randomly select either upper arm or forearm for the target limb to scratch
        # if self.human.gender == 'male':
        #     self.limb, length, radius = [[self.human.right_shoulder, 0.279, 0.043], [self.human.right_elbow, 0.157, 0.033], [self.human.left_shoulder, 0.279, 0.043], [self.human.left_elbow, 0.157, 0.033], [self.human.right_hip, 0.289, 0.078], [self.human.right_knee, 0.172, 0.063], [self.human.left_hip, 0.289, 0.078], [self.human.left_knee, 0.172, 0.063] ][self.np_random.randint(8)]
        # else:
        #     self.limb, length, radius = [[self.human.right_shoulder, 0.264, 0.0355], [self.human.right_elbow, 0.134, 0.027], [self.human.left_shoulder, 0.264, 0.0355], [self.human.left_elbow, 0.134, 0.027], [self.human.right_hip, 0.279, 0.0695], [self.human.right_knee, 0.164, 0.053], [self.human.left_hip, 0.279, 0.0695], [self.human.left_knee,  0.164, 0.053] ][self.np_random.randint(8)]

        if self.human.gender == 'male':
            self.limb, length, radius = [[self.human.right_shoulder, 0.279, 0.043], [self.human.right_elbow, 0.157, 0.033], [self.human.right_hip, 0.289, 0.078], [self.human.right_knee, 0.172, 0.063] ][self.np_random.randint(4)]
        else:
            self.limb, length, radius = [[self.human.right_shoulder, 0.264, 0.0355], [self.human.right_elbow, 0.134, 0.027], [self.human.right_hip, 0.279, 0.0695], [self.human.right_knee, 0.164, 0.053] ][self.np_random.randint(4)]
           
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length*2]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        self.target = self.create_sphere(radius=0.02, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])
        #self.create_sphere(radius=0.1, mass=0.0, pos=arm_pos, visual=True, collision=False, rgba=[0, 0, 1, 1])

        self.update_targets()



    def update_targets(self):
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])



