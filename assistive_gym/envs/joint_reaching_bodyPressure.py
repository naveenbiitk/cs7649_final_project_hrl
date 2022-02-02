import numpy as np
import pybullet as p
import optparse
from scipy.spatial.transform import Rotation as R
import os
import cv2

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture


import sys
sys.path.insert(0,'/nethome/nnagarathinam6/Documents/joint_reaching_evaluation/BodyPressureTRI/networks/')
print(sys.path)
from evaluate_depthREALTIME import Viz3DPose

sys.path.insert(0,'/nethome/nnagarathinam6/Documents/joint_reaching_evaluation/BodyPressureTRI/lib_py/')
from optparse_lib import get_depthnet_options


class JointReachingEnv(AssistiveEnv):

    def __init__(self, robot, human):
        super(JointReachingEnv, self).__init__(robot=robot, human=human, task='joint_reaching', obs_robot_len=(23 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(24 + len(human.controllable_joint_indices)))
        self.blanket_pose_var = False


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
        done = self.iteration >= 200

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

        human_pose_predicted = self.call_bp_cb()
        print("Human pose predicted", shoulder_pos- human_pose_predicted[6] )

        self.total_force_on_human, self.tool_force, self.tool_force_at_target, self.target_contact_pos = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, tool_pos_real - target_pos_real, target_pos_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
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
        joints_positions = [(self.human.j_left_hip_y, -10), (self.human.j_right_hip_y, 10), (self.human.j_left_shoulder_x, -20), (self.human.j_right_shoulder_x, 20)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        #self.human.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2.0, 0, 0])
        self.human.set_base_pos_orient([0, -0.1, 1.1], [-np.pi/2.0, 0, np.pi])

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, self.np_random.uniform(-0.01, 0.01, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        self.prev_target_contact_pos = np.zeros(3)
        # self.robot.print_joint_info()
        self.setup_camera_rpy(camera_target=[0, 0, 0.305 + 2.101], distance=1.5, rpy=[0, -90, 180], fov=50,camera_width=199, camera_height=234)
        # Lock human joints and set velocities to 0
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # * change alpha value so that it is a little more translucent, easier to see the relationship the human
        #p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.75], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.85], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)

        if self.blanket_pose_var:
            delta_y = self.np_random.uniform(-0.05, 0.05)
            delta_x = self.np_random.uniform(-0.02, 0.02)
            delta_rad = self.np_random.uniform(-0.0872665, 0.0872665) # * +/- 5 degrees
            p.resetBasePositionAndOrientation(self.blanket, [0+delta_x, 0.2+delta_y, 1.5], self.get_quaternion([np.pi/2.0, 0, 0+delta_rad]), physicsClientId=self.id)
        else:
            p.resetBasePositionAndOrientation(self.blanket, [0, 0.2, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)


        # * Drop the blanket on the person, allow to settle
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[0.001]*3)

        target_ee_pos = np.array([-0.6, 0, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        #target_ee_orient = np.array([0, 0, 0, 1 ])
        # Not in itch scratch only here
        #self.robot.reset_joints()
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

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

        self.init_bodyPressure()

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


    def init_bodyPressure(self):

        print('----------------------1----------------------')
        p = optparse.OptionParser()
        print('----------------------2----------------------')
        p.add_option('--env', action='store', dest='envse', default='ScratchItchJaco-v1', help='Environment to test (default: ScratchItchJaco-v1)')
        p = get_depthnet_options(p)

        opt,arg = p.parse_args()
        print('----------------------3----------------------')
        print(opt)
        opt.mod=2
        opt.p_idx = 0
        opt.pose_num = 0
        # opt.viz='3D'
        opt.viz = None
        opt.ctype= None
        opt.pimgerr=False

        opt.X_is = 'W'
        opt.slp = 'mixedreal'

        opt.pmr=True
        opt.v2v=True
        opt.no_blanket = False
        

        self.V3D = Viz3DPose(opt)
        self.V3D.load_deep_model()
        self.V3D.load_smpl()

        depth_img = self.get_depth_image()
        #g_t_pose = env.human_pose_smpl_format()
        #print('ground truth: ', g_t_pose)
        #print('image first set',np.mean(img[0:3,0:3]))
        self.calibration_offset = 2121-np.mean(depth_img[0:3,0:3])
        #self.call_bp_cb(depth_img+calibration_offset)
        self.human_pos_offset, self.human_orient_offset = self.human.get_base_pos_orient()
        print('Human pose in assistive gym capsulized body: ')
        print(self.human_pos_offset, self.human_orient_offset)



    def call_bp_cb(self):
        
        #CREATE LOOP HERE
        depth = np.zeros((128, 54)) #this is the depth image.
        depth = self.get_depth_image()+self.calibration_offset
        #print(depth)
        point_cloud = np.zeros((100, 3)) #this is the point cloud. it is optional. You can just set point_cloud=None if you don't want to use it.

        #point_cloud = None
        
        if self.human.gender=='male':
            gen="m"
        else:
            gen="f"

        smpl_verts, human_joints_3D_est, m = self.V3D.estimate_pose(depth, point_cloud=None, project_pressure=False, gender = gen)

        pose_array = np.array(human_joints_3D_est)
        pose_array_reindex = self.pose_reindex(pose_array)
        #print(smpl_verts.shape, human_joints_3D_est.shape)
        #(6890, 3) (25, 3)
        #print('Predicted smpl human pose: ',pose_array)
        return pose_array_reindex


    def human_pose_smpl_format(self):

        agym_jt = np.zeros((20,3))
        #agym_jt
        agym_jt[0,:] = self.human.get_pos_orient(self.human.head)[0]
        agym_jt[1,:] = self.human.get_pos_orient(self.human.neck)[0]
        agym_jt[2,:] = self.human.get_pos_orient(self.human.chest)[0]
        agym_jt[3,:] = self.human.get_pos_orient(self.human.waist)[0]
        agym_jt[4,:] = self.human.get_pos_orient(self.human.upper_chest)[0]
        agym_jt[5,:] = self.human.get_pos_orient(self.human.j_upper_chest_x)[0]
        #arms
        agym_jt[6,:] = self.human.get_pos_orient(self.human.right_upperarm)[0]
        agym_jt[7,:] = self.human.get_pos_orient(self.human.left_upperarm)[0]
        agym_jt[8,:] = self.human.get_pos_orient(self.human.right_forearm)[0]
        agym_jt[9,:] = self.human.get_pos_orient(self.human.left_forearm)[0]
        agym_jt[10,:] = self.human.get_pos_orient(self.human.j_right_wrist_x)[0] 
        agym_jt[11,:] = self.human.get_pos_orient(self.human.j_left_wrist_x)[0]
        agym_jt[12,:] = self.human.get_pos_orient(self.human.right_pecs)[0]
        agym_jt[13,:] = self.human.get_pos_orient(self.human.left_pecs)[0]
        #legs
        agym_jt[14,:] = self.human.get_pos_orient(self.human.right_shin)[0]
        agym_jt[15,:] = self.human.get_pos_orient(self.human.left_shin)[0]
        agym_jt[16,:] = self.human.get_pos_orient(self.human.right_thigh)[0]
        agym_jt[17,:] = self.human.get_pos_orient(self.human.left_thigh)[0]
        agym_jt[18,:] = self.human.get_pos_orient(self.human.right_foot)[0]
        agym_jt[19,:] = self.human.get_pos_orient(self.human.left_foot)[0]

        return agym_jt



    def pose_reindex(self, smpl_pose_jt_1):

        agym_jt_smpl = np.zeros((20,3))

        joints_1 = smpl_pose_jt_1.dot(R.from_euler('x', -90, degrees=True).as_matrix())
        smpl_pose_jt = joints_1.dot(R.from_quat(self.human_orient_offset).as_matrix())

        agym_jt_smpl[0,:] = smpl_pose_jt[15,:] + np.array([ 0.00000000e+00 , 3.72529017e-09, -3.14101279e-02]) 
        agym_jt_smpl[1,:] = smpl_pose_jt[12,:] + np.array([0.00000000e+00 ,1.86264505e-09 ,7.45058060e-09]) 
        agym_jt_smpl[2,:] = smpl_pose_jt[3,:]  + np.array([ 0.00000000e+00 , 3.72529028e-09 ,-2.98023224e-08])
        agym_jt_smpl[3,:] = smpl_pose_jt[0,:]  + np.array([0., 0., 0.])
        agym_jt_smpl[4,:] = smpl_pose_jt[6,:]  + np.array([0.00000000e+00 ,3.72529026e-09 ,2.98023224e-08])
        agym_jt_smpl[5,:] = smpl_pose_jt[9,:]  + np.array([ 0.00000000e+00 ,-5.55111512e-17 ,-7.45058060e-09])
        #arms
        agym_jt_smpl[6,:] = smpl_pose_jt[17,:] + np.array([0.00000000e+00 ,2.79396764e-09 ,0.00000000e+00])
        agym_jt_smpl[7,:] = smpl_pose_jt[16,:] + np.array([ 0.00000000e+00 , 2.79396763e-09 ,-2.98023224e-08])
        agym_jt_smpl[8,:] = smpl_pose_jt[19,:] + np.array([ -0.00457314, -0.0248985 ,  0.02630745])
        agym_jt_smpl[9,:] = smpl_pose_jt[18,:] + np.array([-0.00560603, -0.03206245  ,0.03407168])
        agym_jt_smpl[10,:] = smpl_pose_jt[21,:] + np.array([-0.02053231 ,-0.02895589 ,-0.0012251 ]) 
        agym_jt_smpl[11,:] = smpl_pose_jt[20,:] + np.array([ 0.01074898, -0.02654349  ,0.00240007]) 
        agym_jt_smpl[12,:] = smpl_pose_jt[14,:] + np.array([0.00000000e+00 ,3.72529022e-09 ,7.45058060e-08]) 
        agym_jt_smpl[13,:] = smpl_pose_jt[13,:] + np.array([0.00000000e+00, 3.72529022e-09 ,2.98023224e-08]) 
        #legs
        agym_jt_smpl[14,:] = smpl_pose_jt[5,:] + np.array([ 0.00000000e+00 , 3.72529040e-09 ,-6.24269247e-03])
        agym_jt_smpl[15,:] = smpl_pose_jt[4,:] + np.array([ 0.00000000e+00,  1.86264525e-09 ,-4.67756391e-03])
        agym_jt_smpl[16,:] = smpl_pose_jt[2,:] + np.array([0.00000000e+00 ,3.72529033e-09 ,7.57336617e-04])
        agym_jt_smpl[17,:] = smpl_pose_jt[1,:] + np.array([ 0.00000000e+00 , 1.38777878e-17 ,-4.67753410e-03])
        agym_jt_smpl[18,:] = smpl_pose_jt[8,:] + np.array([ 0.00000000e+00 , 3.72529049e-09 ,-1.32426843e-02])
        agym_jt_smpl[19,:] = smpl_pose_jt[7,:] + np.array([ 0.00000000e+00 , 1.94289029e-16 ,-4.67755646e-03])


        agym_jt_smpl = agym_jt_smpl-agym_jt_smpl[3]+np.array(self.human_pos_offset) #base pose
        
        return  agym_jt_smpl

        #final_error = grnd_truth-pred_final
        #np.linalg.norm(final_error[:,0])/20