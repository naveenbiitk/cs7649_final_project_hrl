import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import human
from .agents.human import Human
from .agents.human_mesh import HumanMesh


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

        # self.human_mesh = HumanMesh()

        # h = self.human_mesh
        # self.body_shape = 'female_1.pkl'
        # self.body_shape = self.np_random.randn(1, self.human_mesh.num_body_shape)

        self.build_assistive_env(furniture_type='hospital_bed',human_impairment='none',fixed_human_base=False)
        self.furniture.set_on_ground()
        self.furniture.set_friction(self.furniture.base, friction=5)

        #self.build_assistive_env(furniture_type=None, human_impairment='none')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        joints_positions = [(self.human.j_right_shoulder_x, 30)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2.0, 0, 0])

        #self.point = self.create_sphere(radius=0.01, mass=0.0, pos=[0, 0, human_height], visual=True, collision=False, rgba=[0, 1, 1, 1])

        p.setGravity(0, 0, -1, physicsClientId=self.id)
        #p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, human_height/2.0], physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)


        # Lock human joints and set velocities to 0
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)


        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'),scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
                              

        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.75], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.blanket, [0, -0.3, 1.5], self.get_quaternion([np.pi / 2.0, 0, 0]),physicsClientId=self.id)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        # Drop the blanket on the person
        for _ in range(30):
            p.stepSimulation(physicsClientId=self.id)

    


        #self.init_env_variables()
        return self._get_obs()

