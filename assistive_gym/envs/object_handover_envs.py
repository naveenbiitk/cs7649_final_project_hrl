from .object_handover import ObjectHandoverEnv
from .agents import stretch, human
from .agents.stretch import Stretch
from .agents.pr2 import PR2
from .agents.human import Human

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm='left'
human_controllable_joint_indices = human.right_arm_joints


class ObjectHandoverStretchEnv(ObjectHandoverEnv):
	def __init__(self):
		super(ObjectHandoverStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
		
class ObjectHandoverStretchHumanEnv(ObjectHandoverEnv, MultiAgentEnv):
    def __init__(self):
        super(ObjectHandoverStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ObjectHandoverStretchHuman-v1', lambda config: ObjectHandoverStretchHumanEnv())

class ObjectHandoverPR2Env(ObjectHandoverEnv):
	def __init__(self):
		super(ObjectHandoverPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
		
class ObjectHandoverPR2HumanEnv(ObjectHandoverEnv, MultiAgentEnv):
    def __init__(self):
        super(ObjectHandoverPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ObjectHandoverPR2Human-v1', lambda config: ObjectHandoverPR2HumanEnv())

