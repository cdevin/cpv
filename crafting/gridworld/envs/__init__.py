from gym.envs.registration import register
from gridworld.policies.gridworld_policies import  policies, policy_names, policy_dict, policy_inputs, policy_outputs, TASK_EVAL_MAP

for name in policy_names:
    register(
        id='HammerWorld-'+name+'-v0',
        entry_point='gridworld.envs.grid_affordance:HammerWorld',
        kwargs={'res':3, 'add_objects':policy_inputs[name], 'visible_agent':True,
                'agent_centric':True, 'success_function':  TASK_EVAL_MAP[name],
                'reward_function': TASK_EVAL_MAP[name],
        }
        )
    register(
        id='HammerWorld-StateObs-'+name+'-v0',
        entry_point='gridworld.envs.grid_affordance:HammerWorld',
        kwargs={'res':3, 'add_objects':policy_inputs[name], 'visible_agent':True,
                'reward_function': TASK_EVAL_MAP[name], 'state_obs':True,
                'success_function':  TASK_EVAL_MAP[name],
        }
        )

##################
# Language Gym Registration
##################
register(
    id='Crafting-Atomic-v0',
    entry_point='gridworld.envs.babayai_wrapper:GymCraftingEnv',
    kwargs={'num_tasks': 1}
)
register(
    id='Crafting-2Tasks-v0',
    entry_point='gridworld.envs.babayai_wrapper:GymCraftingEnv',
    kwargs={'num_tasks': 2}
)
register(
    id='Crafting-3Tasks-v0',
    entry_point='gridworld.envs.babayai_wrapper:GymCraftingEnv',
    kwargs={'num_tasks': 3}
)
register(
    id='Crafting-4Tasks-v0',
    entry_point='gridworld.envs.babayai_wrapper:GymCraftingEnv',
    kwargs={'num_tasks': 4}
)