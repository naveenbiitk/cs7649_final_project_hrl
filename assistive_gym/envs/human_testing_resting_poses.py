import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human

human_controllable_joint_indices = human.right_arm_joints + human.left_arm_joints
class HumanTestingEnv(AssistiveEnv):
    def __init__(self):
        super(HumanTestingEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=False), task='human_testing', obs_robot_len=0, obs_human_len=0)

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        
        self.take_step(action, gains=0.05, forces=1.0)
        return [], 0, False, {}

    def _get_obs(self, agent=None):
        return []

    def reset(self):
        super(HumanTestingEnv, self).reset()
        self.build_assistive_env(furniture_type='wheelchair',human_impairment='none',fixed_human_base=False)
        self.furniture.set_on_ground()
        self.furniture.set_friction(self.furniture.base, friction=5)

        #self.build_assistive_env(furniture_type=None, human_impairment='none')

        # Set joint angles for human joints (in degrees)
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        human_height, human_base_height = self.human.get_heights()
        #print('Human height:', human_base_height, 'm')
        
        chair_seat_position = np.array([0, -0.05, 0.6])
        self.human.set_base_pos_orient([0, -0.1, human_base_height+0.2], [0, 0, 0, 1])
        #print('Printing pose ',chair_seat_position - self.human_mesh.get_vertex_positions(self.human_mesh.bottom_index))
        #self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])

        joints_positions = [(self.human.j_right_elbow, -115),(self.human.j_right_shoulder_x, 130), (self.human.j_right_shoulder_z, -100), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, -35), (self.human.j_head_y, -35), (self.human.j_neck, -30 ) ]

        #joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -180), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        #joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        #joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None, reactive_gain=0.01)


        #self.point = self.create_sphere(radius=0.01, mass=0.0, pos=[0, 0, human_height], visual=True, collision=False, rgba=[0, 1, 1, 1])

        
        #p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, human_height/2.0], physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        #self.init_env_variables()
        return self._get_obs()

