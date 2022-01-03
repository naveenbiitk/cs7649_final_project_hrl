from .joint_reaching import JointReachingEnv
from .agents import stretch, human
from .agents.stretch import Stretch
from .agents.pr2 import PR2
from .agents.human import Human

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm='left'
human_controllable_joint_indices = human.right_arm_joints+human.right_leg_joints


class JointReachingStretchEnv(JointReachingEnv):
	def __init__(self):
		super(JointReachingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
		
class JointReachingStretchHumanEnv(JointReachingEnv, MultiAgentEnv):
    def __init__(self):
        super(JointReachingStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:JointReachingStretchHuman-v1', lambda config: JointReachingStretchHumanEnv())

class JointReachingPR2Env(JointReachingEnv):
	def __init__(self):
		super(JointReachingPR2Env, self).__init__(robot=PR2('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
		
class JointReachingPR2HumanEnv(JointReachingEnv, MultiAgentEnv):
    def __init__(self):
        super(JointReachingPR2HumanEnv, self).__init__(robot=PR2('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:JointReachingPR2Human-v1', lambda config: JointReachingPR2HumanEnv())