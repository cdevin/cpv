import gym
import numpy as np
import sys
from six import StringIO, b
import copy
from gym import Env
from gridworld.envs.grid_affordance import HammerWorld, OBJECTS
from gridworld.policies.language import simple_language
from gridworld.policies.composite_policy import CompositePolicy

policy_diffs = {
    'EatBreadPolicy': {'bread': -1},
    'ChopTreePolicy': {'tree': -1, 'sticks': 1},
    'BuildHousePolicy': {'house': 1, 'sticks': -1},
    'MakeBreadPolicy': {'bread': 1, 'wheat': -1},
    'ChopRockPolicy': {'rock': -1},
    }

def get_true_counts_diff(policy_list):
    counts_diff = {obj: [] for obj in OBJECTS}
    for obj in OBJECTS:
        for pol in policy_list:
            if obj in policy_diffs[pol]:
                counts_diff[obj].append(policy_diffs[pol][obj])
    return counts_diff

def eval_counts(ref_counts_diff, counts_diff):
    counts_diff = copy.deepcopy(counts_diff)
    success =1
    for obj in ref_counts_diff.keys():
        for t, diff in enumerate(ref_counts_diff[obj]):
            if diff != 0:
                if diff in counts_diff[obj]:
                    i = counts_diff[obj].index(diff)
                    counts_diff[obj][counts_diff[obj].index(diff)] = 0
                else:
                    success = 0
    return success

def get_counts_diff(counts):
    return {obj: [si-sj for si,sj in zip(counts[obj][1:],counts[obj][:-1] )] for obj in counts.keys()}

class GymCraftingEnv(Env):
    """
    A Gym API to the crafting environment.
    """
    def __init__(self, num_tasks, **kwargs ):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.env = HammerWorld(add_objects =[],res=1, visible_agent=True, use_exit=False, 
                               agent_centric=False, goal_dim=0, size=[10, 10],
                               few_obj=False, render_mode='one_hot')
        self.policy_env = HammerWorld(add_objects =[],res=1, visible_agent=True, use_exit=False, 
                               agent_centric=False, goal_dim=0, size=[10, 10],
                               few_obj=False, render_mode='one_hot')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        instruction, policy_list = simple_language(self.num_tasks)
        self.instruction = instruction
        policy = CompositePolicy(policy_list, self.env.action_space, self.policy_env, noise_level=0.1)
        self.policy = policy
        self.policy_list = policy_list
        self.policy.reset()
        self.ref_counts_diff = get_true_counts_diff(policy_list)
        obs = self.env.reset(min_obj = self.policy.min_object_nums())
        self.states = [copy.deepcopy(self.env.state)]
        obs = np.reshape(obs, (11,10,9))
        return {'image': obs, 'instr': instruction}
        
    def step(self, a):
        obs, r, d, info = self.env.step(a)
        self.states.append(copy.deepcopy(self.env.state))
        object_counts = {obj: [s['object_counts'][obj] for s in self.states] for obj in OBJECTS}
        counts_diff = get_counts_diff(object_counts)
        success = eval_counts(self.ref_counts_diff, counts_diff)
        info['oject_counts'] = counts_diff
        info['reference_counts'] = self.ref_counts_diff
        info['policy_list'] = self.policy_list
        if success:
            d = True
            r = 1
        obs = np.reshape(obs, (11,10,9))
        return {'image': obs, 'mission': self.instruction}, r, d, info
    
    