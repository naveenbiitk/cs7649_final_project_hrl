from .joint_motion import JointMotionEnv
from .agents import stretch, human
from .agents.stretch import Stretch
from .agents.pr2 import PR2
from .agents.human import Human

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm='left'
human_controllable_joint_indices = human.right_arm_joints

class JointMotionStretchEnv(JointMotionEnv):
	def __init__(self):
		super(JointMotionStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
		
class JointMotionStretchHumanEnv(JointMotionEnv, MultiAgentEnv):
    def __init__(self):
        super(JointMotionStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:JointMotionStretchHuman-v1', lambda config: JointMotionStretchHumanEnv())

class JointMotionPR2Env(JointMotionEnv):
	def __init__(self):
		super(JointMotionPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
		
class JointMotionPR2HumanEnv(JointMotionEnv, MultiAgentEnv):
    def __init__(self):
        super(JointMotionPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:JointMotionPR2Human-v1', lambda config: JointMotionPR2HumanEnv())

