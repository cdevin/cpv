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


register(
    id='HammerWorld-v0',
    entry_point='gridworld.envs.grid_affordance:HammerWorld',
    kwargs={'res':3, 'visible_agent':False}
)

def placeholder_reward(init_state, final_state):
    raise NotImplementedError


register(
    id='HammerWorld-PlaceholderReward-v0',
    entry_point='gridworld.envs.grid_affordance:HammerWorld',
    kwargs={'res':3, 'visible_agent':True, 'agent_centric':True,
            'reward_function':placeholder_reward, 'batch_reward':True}
)
for name in policy_names:
    register(
        id='HammerWorld-PlaceholderReward-'+name+'-v0',
        entry_point='gridworld.envs.grid_affordance:HammerWorld',
        kwargs={'res':3, 'visible_agent':True, 'agent_centric':True, 
                'reward_function':placeholder_reward, 'batch_reward':True,
                'success_function':  TASK_EVAL_MAP[name], 'add_objects':policy_inputs[name],
               }
    )
    
for name in policy_names:
    register(
        id='HammerWorld-Delta-'+name+'-v0',
        entry_point='gridworld.envs.grid_affordance:HammerWorld',
        kwargs={'res':3, 'visible_agent':True, 'agent_centric':False,
                'reward_function':TASK_EVAL_MAP[name], 
                'success_function':  TASK_EVAL_MAP[name], 'add_objects':policy_inputs[name],
                'goal_dim': 64,
               }
    )
register(
    id='ResettingWorld-v0',
    entry_point='gridworld.envs.resetting_affordance:ResettingHammerWorld',
    kwargs={'res':3, 'visible_agent':True, 'agent_centric':False, 'success_function':  TASK_EVAL_MAP['EatBreadPolicy'],
            'reward_function':TASK_EVAL_MAP['EatBreadPolicy'], 'batch_reward':False}
)
register(
    id='GoalAffordanceWorld-v0',
    entry_point='gridworld.envs.goalenv_grid_affordance:GoalHammerWorld',
    kwargs={'res':3, 'visible_agent':True, 'agent_centric':False,'batch_reward':False}
)

register(
    id='GoalResettingWorld-v0',
    entry_point='gridworld.envs.goal_resetting_affordance:GoalResettingHammerWorld',
    kwargs={'res':3, 'visible_agent':True, 'agent_centric':False, 'batch_reward':False}
)
register(
    id='IBCGoalResettingWorld-v0',
    entry_point='gridworld.envs.ibc_world:IBCGoalResettingHammerWorld',
    kwargs={'res':3, 'visible_agent':True, 'agent_centric':False, 'batch_reward':False}
)