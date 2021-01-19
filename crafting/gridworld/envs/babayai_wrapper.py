
import numpy as np
import sys
from six import StringIO, b
import copy
from gym import Env
from gridworld.envs.grid_affordance import HammerWorld, OBJECTS
from gridworld.policies.language import simple_language

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
                counts_diff[obj].append(policy_diffs[pol])
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
    def reset(self):
        instruction, policy_list = simple_language(self.num_tasks)
        policy = CompositePolicy(policy_list, self.env.action_space, self.policy_env, noise_level=0.1)
        self.ref_counts_diff = get_true_counts_diff(policy_list)
        obs = self.env.reset(min_obj = policy.min_object_nums())
        self.states = [copy.deepcopy(self.env.state)]
        return obs
        
    def step(self, a):
        obs, r, d, info = self.env.step(a)
        self.states.append(copy.deepcopy(self.env.state))
        object_counts = {obj: [s['object_counts'][obj] for s in self.states] for obj in OBJECTS}
        counts_diff = get_counts_diff(object_counts)
        success = eval_counts(self.ref_counts_diff, counts_diff)
        if success:
            d = True
            r = 1
        return obs, r, d, info
    
    
##################
# Gym Registration
##################

gym.envs.registration.register(
    id='Crafting-Atomic-v0',
    entry_point='gridworld.envs.grid_affordance.babayai_wrapper:GymCraftingEnv',
    kwargs={'num_tasks': 1}
)
gym.envs.registration.register(
    id='Crafting-2Tasks-v0',
    entry_point='gridworld.envs.grid_affordance.babayai_wrapper:GymCraftingEnv',
    kwargs={'num_tasks': 2}
)
gym.envs.registration.register(
    id='Crafting-3Tasks-v0',
    entry_point='gridworld.envs.grid_affordance.babayai_wrapper:GymCraftingEnv',
    kwargs={'num_tasks': 3}
)
gym.envs.registration.register(
    id='Crafting-4Tasks-v0',
    entry_point='gridworld.envs.grid_affordance.babayai_wrapper:GymCraftingEnv',
    kwargs={'num_tasks': 4}
)