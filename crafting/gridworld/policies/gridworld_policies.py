# TODO take best path
import numpy as np
import math
import random
import collections 

#from rlkit.policies.base import SerializablePolicy

from gridworld.envs.grid_affordance import HammerWorld, ACTIONS
import copy
# 0: UP
# 1: RIGHT
# 2: DOWN
# 3: LEFT
ACTION_CONVERTER = {v: k for k,v in enumerate(ACTIONS)}
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return x

class BaseGridStatePolicy():
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space, env, noise_level=0.0):
        self.action_space = action_space
        self.env = env
        self.action_queue = []
        self.success = False
        self.noise_level = noise_level

    def reset(self):
        self.action_queue = []

    def get_nearest_square(self, agent_pos, possible_squares=None, impossible_squares=None):
        distances = []
        if possible_squares is not None:
            for x in range(self.env.nrow):
                for y in range(self.env.ncol):
                    if (x,y) in possible_squares:
                        distances.append((abs(agent_pos[0]-x)+abs(agent_pos[1]-y), (x,y)))
        else:
            for x in range(self.env.nrow):
                for y in range(self.env.ncol):
                    if (x,y) not in impossible_squares:
                        distances.append((abs(agent_pos[0]-x)+abs(agent_pos[1]-y), (x,y)))
        distances = sorted(distances)
        target_pos = distances[0][1]
        return target_pos

    def get_random_square(self, agent_pos, possible_squares=None, impossible_squares=None):

        if possible_squares is None:
            possible_squares = []
            for x in range(self.env.nrow):
                for y in range(self.env.ncol):
                    if (x,y) not in impossible_squares:
                        possible_squares.append((x,y))
        target = random.randint(0, len(possible_squares)-1)
        target_pos = possible_squares[target]
        #print("drop target_pos", target_pos)
        return target_pos
    
    def state2grid(self, state, impassable_objects):
        grid = np.zeros((self.env.nrow, self.env.ncol))
        #import pdb; pdb.set_trace()
        for obj, pos in state['object_positions'].items():
            #print("obj", obj, pos)
            if '_' in obj:
                root =  obj.split('_')[0]
                if root in impassable_objects:
                    #print("objc is impassable", pos)
                    grid[pos[0], pos[1]] = 1
        return grid
    
    def get_impassable_objects(self, state):
        objects = []
        if state['holding'] == '':
            objects += ['rock', 'tree']
        elif state['holding'].startswith('axe'):
            objects.append('rock')
        elif state['holding'].startswith('hammer'):
            objects.append('tree')
        return objects
    
    def BFS(self, agent_pos, target_pos, grid):
        """
        agent_pos: tuple of (x,y) agent position
        target_pos: tuple of (x,y) target position
        grid: numpy array of 0s in traversable space and 1s in free space
        """
        wall = 1
        queue = collections.deque([[agent_pos]])
        seen = set([agent_pos])
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x,y) == target_pos:
                return path
            action_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            random.shuffle(action_list)
            for action in action_list:
                x2, y2 = x+action[0], y+action[1]
                if 0 <= x2 < grid.shape[0] and 0 <= y2 < grid.shape[1] and grid[x2][y2] != wall and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))
    
    def plot_path(self, agent_pos, target_pos, state, impassable_objects, optional_impassables=[]):
        path = None
        while path is None and len(optional_impassables) > 0:
            grid = self.state2grid(state, impassable_objects + optional_impassables)
            noise_level = self.noise_level
            path = self.BFS(agent_pos, target_pos, grid)
            if path is None:
                optional_impassables = optional_impassables[:-1]
        if path is None:
            actions = []  
        else:
            actions_raw  = [(sign(pos[0]-pre_pos[0]),sign(pos[1]-pre_pos[1])) for
                               pos, pre_pos  in zip(path[1:], path[:-1])]
            actions = [ACTION_CONVERTER[act] for act in actions_raw]
        return actions, path, path is not None



    def get_action(self, state):
        pass



class DropObjectPolicy(BaseGridStatePolicy):
    """Policy goes to a nearby empty square and drops object"""
    def __init__(self, *args, nearest=True, except_object=None, noise_level=0):
        super().__init__(*args, noise_level=noise_level)
        self.nearest = nearest
        self.except_object = except_object
        self.move_over_pol = MoveOverPolicy(*args, noise_level=noise_level)
        self.last_action = None
        
    def get_action(self, state):
        #print("running drop pol", self.nearest, "nearest")
        if len(self.action_queue) == 0:
            #print("drop pol has no queue")
            
            if len(state['holding']) == 0:  
                #print("drop pol has no hold")
                return None
            if self.except_object is not None:
                if state['holding'].startswith(self.except_object):
                    return None

            agent_pos = state['agent']
            distances = []
            nonempty_squares = []
            held_obj = state['holding']
            for obj, pos in state['object_positions'].items():
                nonempty_squares.append(pos)
            if len(nonempty_squares) == self.env.nrow*self.env.ncol:
                print("All squares are full")
                return None
            if self.nearest:
                target_pos = self.get_nearest_square(agent_pos, impossible_squares=nonempty_squares)
            else:
                target_pos = self.get_random_square(agent_pos, impossible_squares=nonempty_squares+[agent_pos])
            impassable_objects = self.get_impassable_objects(state)
            optional_impassables = ['bread', 'sticks', 'wheat']
            out = self.plot_path(agent_pos, target_pos, state, impassable_objects, optional_impassables)

            actions, states, good = out
            self.success = good
            if not good:
                return ACTION_CONVERTER['exit']
            self.action_queue = actions + [ACTION_CONVERTER['drop'],random.randint(0,3), None]
            a = self.action_queue[0]
            self.last_action = a
            self.action_queue = self.action_queue[1:]
            if len(self.action_queue) == 0 and self.last_action == ACTION_CONVERTER['drop']:
                self.action_queue.append(self.move_over_pol.get_action(state))
                self.action_queue.append(None)
        else:
            a = self.action_queue[0]
            self.action_queue = self.action_queue[1:]
            self.last_action = a
        return a

class MoveOverPolicy(BaseGridStatePolicy):
    def get_action(self):
        if len(self.action_queue) == 0:
            agent_pos = state['agent']
            obj_positions = state.obj_positions.values()
            target_pos = self.get_nearest_square(agent_pos, impossible_squares=[agent_pos]+obj_positions)
            impassable_objects = self.get_impassable_objects(state)
            optional_impassables = ['bread', 'sticks', 'wheat']
            out = self.plot_path(agent_pos, target_pos, state, impassable_objects, optional_impassables)

            # print("Got", out)
            #import pdb; pdb.set_trace()
            actions, states, good = out
            self.action_queue = actions + [None]
            a = self.action_queue[0]
            self.action_queue = self.action_queue[1:]
        else:
            #print("move pol has queue", self.action_queue)
            a = self.action_queue[0]
            self.action_queue = self.action_queue[1:]
        return a

class GoToCornerPolicy(BaseGridStatePolicy):
    def get_action(self):
        if len(self.action_queue) == 0:
            agent_pos = state['agent']
            target_pos = (0,0)#self.get_nearest_square(agent_pos, impossible_squares=[agent_pos])
            impassable_objects = self.get_impassable_objects(state)# + ['bread', 'sticks', 'wheat']
            out = self.plot_path(agent_pos, target_pos, state, impassable_objects)
            # print("Got", out)
            #import pdb; pdb.set_trace()
            actions, states, good = out
            self.action_queue = actions + [None]
            a = self.action_queue[0]
            self.action_queue = self.action_queue[1:]
        else:
            #print("move pol has queue", self.action_queue)
            a = self.action_queue[0]
            self.action_queue = self.action_queue[1:]
        return a


class GoToObjectPolicy(BaseGridStatePolicy):
    """ Policy picks up given object"""
    def assign_obj(self, obj_name):
        self.target_obj = obj_name

    def get_action(self, state):
        if len(self.action_queue) == 0:
            agent_pos = state['agent']
            obj_pos = None
            obj_squares = []
            for obj in state['object_positions'].keys():
                if obj.startswith(self.target_obj):
                    obj_squares.append(state['object_positions'][obj])

            if len(obj_squares) == 0:
                return ACTION_CONVERTER['exit']
            obj_pos =  self.get_nearest_square(agent_pos, possible_squares=obj_squares)
            impassable_objects = self.get_impassable_objects(state) 
            optional_impassables = []
            for obj in  ['bread', 'sticks', 'wheat']:
                if not obj.startswith(self.target_obj):
                    optional_impassables.append(obj)

            out = self.plot_path(agent_pos, obj_pos, state, impassable_objects, optional_impassables)

            actions, states, good = out

            if not good or len(states) > 0 and states[-1] != obj_pos:


                return ACTION_CONVERTER['exit']
                #import pdb; pdb.set_trace()
            self.action_queue = actions
            if len(self.action_queue) < 1 or not good:
                return None
            a = self.action_queue[0]
            self.action_queue = self.action_queue[1:]
        else:
            a = self.action_queue[0]
            self.action_queue = self.action_queue[1:]
            if len(self.action_queue) == 0 and a is not None:
                self.action_queue.append(None)
        return a

class EatBreadPolicy(GoToObjectPolicy):
    def __init__(self, *args, noise_level=0):
        super().__init__(*args, noise_level=noise_level)
        self.target_obj = 'bread'
    def get_action(self, state):
        a = super().get_action(state)
        if a is None:
            a = np.random.randint(4)
            self.action_queue.append(ACTION_CONVERTER['exit'])
        return a

class SimplePickupObjectPolicy(GoToObjectPolicy):
    def __init__(self, *args, noise_level=0):
        super().__init__(*args, noise_level=noise_level)
        #self.started = False

    def get_action(self, state):
        if state['holding'].startswith(self.target_obj):
            return None
        if len(self.action_queue) == 0:
            # print("pickup pol is empty, get a path")
            a = super().get_action(state)
            self.action_queue.append(ACTION_CONVERTER['pickup'])
            self.action_queue.append(None)
        else:

            a = super().get_action(state)
        return a

class PickupObjectPolicy(BaseGridStatePolicy):
    def __init__(self, *args, target_obj='axe', noise_level=0):
        super().__init__(*args, noise_level=noise_level)
        self.target_obj = target_obj
        self.pickup_obj_pol =  SimplePickupObjectPolicy(*args, noise_level=noise_level)
        self.drop_obj_pol_near = DropObjectPolicy(*args, nearest=True, except_object=target_obj, noise_level=noise_level)
        self.pickup_obj_pol.target_obj = target_obj
        self.policy_queue = [self.drop_obj_pol_near, self.pickup_obj_pol]

    def reset(self):
        self.pickup_obj_pol.reset()
        self.drop_obj_pol_near.reset()
        self.action_queue = []
        self.policy_queue = [self.drop_obj_pol_near, self.pickup_obj_pol]

    def get_action(self, state):

        a = self.policy_queue[0].get_action(state)
        if a is None:
            self.policy_queue[0].reset()
            if len(self.policy_queue) == 1:
                a = ACTION_CONVERTER['exit']

            else:
                self.policy_queue = self.policy_queue[1:]
                a = self.policy_queue[0].get_action(state)
        if a == ACTION_CONVERTER['exit']:
            self.reset()
        return a

class MoveObjectPolicy(BaseGridStatePolicy):
    def __init__(self, *args, target_obj='axe', noise_level=0):
        super().__init__(*args, noise_level=noise_level)
        self.target_obj = target_obj
        self.pickup_obj_pol =  PickupObjectPolicy(*args, target_obj=target_obj, noise_level=noise_level)
        self.drop_obj_pol_random = DropObjectPolicy(*args, nearest=False, noise_level=noise_level)
        #self.pickup_obj_pol.target_obj = target_obj
        self.policy_queue = [ self.pickup_obj_pol, self.drop_obj_pol_random]

    def reset(self):
        self.pickup_obj_pol.reset()
        self.drop_obj_pol_random.reset()
        self.action_queue = []
        self.policy_queue = [ self.pickup_obj_pol, self.drop_obj_pol_random]

    def get_action(self, state):
        # print("state before", state)
        # new_state = copy.deepcopy(state)
        
        #import pdb; pdb.set_trace()
        a = self.policy_queue[0].get_action(state)
        if a is None or (a == 6 and len(self.policy_queue) > 1):
            self.policy_queue[0].reset()
            if len(self.policy_queue) == 1:
                a = ACTION_CONVERTER['exit']
                #self.reset()
                #return a
            else:
                self.policy_queue = self.policy_queue[1:]
                a = self.policy_queue[0].get_action(state)
        if a == ACTION_CONVERTER['exit']:
            self.reset()
        return a

class ChopTreePolicy(BaseGridStatePolicy):
    def __init__(self, *args, noise_level=0):
        super().__init__(*args, noise_level=noise_level)
        #self.drop_obj_pol = DropObjectPolicy(*args)
        self.pickup_obj_pol = PickupObjectPolicy(*args, target_obj = 'axe', noise_level=noise_level)
        self.goto_obj_pol = GoToObjectPolicy(*args, noise_level=noise_level)
        self.pickup_obj_pol.target_obj = 'axe'
        self.goto_obj_pol.target_obj = 'tree'
        self.policy_queue = [self.pickup_obj_pol, self.goto_obj_pol]

    def reset(self):
        self.pickup_obj_pol.reset()
        self.goto_obj_pol.reset()
        self.action_queue = []
        self.policy_queue = [self.pickup_obj_pol, self.goto_obj_pol]

    def get_action(self, state):
        # print("state before", state)
        #new_state = copy.deepcopy(state)
        a = self.policy_queue[0].get_action(state)
        if a is None or (a == 6 and len(self.policy_queue) > 1):
            #self.policy_queue[0].reset()
            if len(self.policy_queue) == 1:
                a = np.random.randint(4)
                self.policy_queue[0].action_queue.append(ACTION_CONVERTER['exit'])
                #self.reset()
                #return a
            else:
                self.policy_queue = self.policy_queue[1:]
                a = self.policy_queue[0].get_action(state)

        return a


class BuildHousePolicy(ChopTreePolicy):
    def __init__(self, *args, noise_level=0):
        self.pickup_obj_pol = PickupObjectPolicy(*args, target_obj = 'hammer', noise_level=noise_level)
        self.goto_obj_pol = GoToObjectPolicy(*args, noise_level=noise_level)
#         self.pickup_obj_pol.target_obj = 'axe'
#         self.goto_obj_pol.target_obj = 'tree'
        self.policy_queue = [self.pickup_obj_pol, self.goto_obj_pol]
        self.pickup_obj_pol.target_obj = 'hammer'
        self.goto_obj_pol.target_obj = 'sticks'
        
class MakeBreadPolicy(ChopTreePolicy):
    def __init__(self, *args, noise_level=0):
        self.pickup_obj_pol = PickupObjectPolicy(*args, target_obj = 'axe', noise_level=noise_level)
        self.goto_obj_pol = GoToObjectPolicy(*args, noise_level=noise_level)
#         self.pickup_obj_pol.target_obj = 'axe'
#         self.goto_obj_pol.target_obj = 'tree'
        self.policy_queue = [self.pickup_obj_pol, self.goto_obj_pol]
        self.pickup_obj_pol.target_obj = 'axe'
        self.goto_obj_pol.target_obj = 'wheat'

class Random():
    def __init__(self,  action_space, envs, noise_level=0):
        self.num_actions = action_space
        

    def get_action(self, state):
        agent_pos = state['agent']
        if len(state['holding']) == 0:
            for obj, pos in state['object_positions'].items():
                if pos == agent_pos:
                    if obj.startswith('axe') or obj.startswith('hammer'):
                        pickup = np.random.randint(2)
                        if pickup:
                            return ACTION_CONVERTER['pickup']
        return np.random.randint(6)
    
    def reset(self):
        pass


class ChopRockPolicy(ChopTreePolicy):
    def __init__(self, *args, noise_level=0):
        self.pickup_obj_pol = PickupObjectPolicy(*args, target_obj = 'hammer', noise_level=noise_level)
        self.goto_obj_pol = GoToObjectPolicy(*args, noise_level=noise_level)

        self.policy_queue = [self.pickup_obj_pol, self.goto_obj_pol]
        self.pickup_obj_pol.target_obj = 'hammer'
        self.goto_obj_pol.target_obj = 'rock'
def PickupAxePolicy(action, env, noise_level=0):
    pol = MoveObjectPolicy(action, env, target_obj ='axe', noise_level=noise_level)
    return pol
def PickupHammerPolicy(action, env, noise_level=0):
    pol = MoveObjectPolicy(action, env, target_obj='hammer', noise_level=noise_level)
    return pol
def PickupSticksPolicy(action, env, noise_level=0):
    pol = MoveObjectPolicy(action, env, target_obj='sticks', noise_level=noise_level)
    return pol

class GoToHousePolicy(GoToObjectPolicy):
    def __init__(self, *args, noise_level=0):
        super().__init__(*args,  noise_level=0)
        self.target_obj = 'house'
    def get_action(self, state):
        a = super().get_action(state)
        if a is None:
            a = np.random.randint(4)
            #self.action_queue.append(direction)
            self.action_queue.append(ACTION_CONVERTER['exit'])
        return a



policies = [EatBreadPolicy, GoToHousePolicy, PickupHammerPolicy, PickupAxePolicy, PickupSticksPolicy, ChopTreePolicy, BuildHousePolicy, ChopRockPolicy, MakeBreadPolicy, Random]
policy_names = ['EatBreadPolicy', 'GoToHousePolicy', 'PickupHammerPolicy', 'PickupAxePolicy', 'ChopTreePolicy', 'BuildHousePolicy', 'ChopRockPolicy', 'PickupSticksPolicy', 'MakeBreadPolicy', 'random']

policy_dict = {k:v for k,v in zip(policy_names, policies)}
policy_inputs = {
    'EatBreadPolicy': ['bread'],
    'GoToHousePolicy': ['house'],
    'PickupHammerPolicy': ['hammer'],
    'PickupAxePolicy': ['axe'],
    'ChopTreePolicy': ['axe', 'tree'],
    'BuildHousePolicy': ['hammer', 'sticks'],
    'ChopRockPolicy': ['hammer', 'rock'],
    'PickupSticksPolicy': ['sticks'],
    'MakeBreadPolicy': ['wheat', 'axe'],
    'GoToCornerPolicy': [],
    'random': [],
    }
policy_outputs = {
    'EatBreadPolicy': [],# ['bread'],
    'GoToHousePolicy': [],#,['house'],
    'PickupHammerPolicy': [],# ['hammer'],
    'PickupAxePolicy': [],#['axe'],
    'ChopTreePolicy': ['sticks'],
    'BuildHousePolicy': ['house'],
    'PickupSticksPolicy': [],
    'MakeBreadPolicy': ['bread'],
    'ChopRockPolicy': [],
    'GoToCornerPolicy': [],
    }
policy_removes = {
    'EatBreadPolicy': ['bread'],
    'GoToHousePolicy': [],#,['house'],
    'PickupHammerPolicy': [],# ['hammer'],
    'PickupAxePolicy': [],#['axe'],
    'ChopTreePolicy': ['tree'],
    'BuildHousePolicy': ['sticks'],
    'ChopRockPolicy': ['rock'],
    'PickupSticksPolicy': [],
    'MakeBreadPolicy': ['wheat'],
    'GoToCornerPolicy': []
    }
def eval_eatbread(init_state, final_state):
    success=  init_state['object_counts']['bread']>final_state['object_counts']['bread'] 
#     if not success:
#         print("EatBreadPol:", init_state['object_counts']['bread'],final_state['object_counts']['bread'] )
    return success

def eval_choptree(init_state, final_state):
    success=  final_state['object_counts']['tree'] < init_state['object_counts']['tree']
#     if not success:
#         print("ChopTree:", init_state['object_counts']['tree'],final_state['object_counts']['tree'] )
    return success

def eval_choprock(init_state, final_state):
    success=  final_state['object_counts']['rock'] < init_state['object_counts']['rock']
#     if not success:
#         print("ChopRock:", init_state['object_counts']['rock'],final_state['object_counts']['rock'] )
    return success

def eval_buildhouse(init_state, final_state):
    #print("buildhouse, num_before", init_state['object_counts']['house'], "num after",final_state['object_counts']['house'] )
    success=  final_state['object_counts']['house'] > init_state['object_counts']['house']
#     if not success:
#         print("BuildHouse:", init_state['object_counts']['house'],final_state['object_counts']['house'] )
    return success

def eval_pickupaxe(init_state, final_state):
    for obj in init_state['object_positions'].keys():
        if obj.startswith('axe'):
            if final_state['object_positions'][obj] != init_state['object_positions'][obj]:
                success=  True
                return success
    success=  False
#     if not success:
#         print("PickupAxe:",  init_state['object_positions'],final_state['object_positions'] )
    return success

def eval_pickuphammer(init_state, final_state):
    for obj in init_state['object_positions'].keys():
        if obj.startswith('hammer'):
            if final_state['object_positions'][obj] != init_state['object_positions'][obj]:
                success=  True
                return success
    success=  False
#     if not success:
#         print("PickupHammer:",  init_state['object_positions'],final_state['object_positions']  )
    return success

def eval_gotohouse(init_state, final_state):
    success=  final_state['hunger'] <1.0
#     if not success:
#         print("GoToHouse:", init_state['hunger'],final_state['hunger'] )
    return success

def eval_pickupsticks(init_state, final_state):
    for obj in init_state['object_positions'].keys():
        if obj.startswith('sticks'):
            if obj in  final_state['object_positions']:
                if final_state['object_positions'][obj] != init_state['object_positions'][obj]:
                    success=  True
                    return success
    success=  False
#     if not success:
#         print("PickupAxe:",  init_state['object_positions'],final_state['object_positions'] )
    return success

def eval_makebread(init_state, final_state):
    success=final_state['object_counts']['wheat'] < init_state['object_counts']['wheat']
    return success

def eval_gotocorner(init_state, final_state):
    agent_pos = final_state['agent']
    for obj in init_state['object_positions'].keys():
        if obj.startswith('house'):
            if final_state['object_positions'][obj] == agent_pos:
                return True
#     if not success:
#         print("GoToCorner",final_state['agent'] )
    return False

TASK_EVAL_MAP = {'EatBreadPolicy': eval_eatbread,
                 'ChopTreePolicy': eval_choptree,
                 'BuildHousePolicy': eval_buildhouse,
                 'ChopRockPolicy': eval_choprock,
                 'PickupAxePolicy': eval_pickupaxe,
                 'PickupHammerPolicy': eval_pickuphammer,
                 'GoToHousePolicy': eval_gotohouse,
                 'PickupSticksPolicy': eval_pickupsticks,
                 'MakeBreadPolicy': eval_makebread,
                 'GoToCornerPolicy': eval_gotocorner,
                 'random': lambda x,y: True,
}


