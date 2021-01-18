import numpy as np

from gym import Env, spaces
from gym.utils import seeding

import sys
from six import StringIO, b
import copy
from gym import utils
from gym.envs.toy_text import discrete
from scipy.misc import imresize 
RENDER_DIR = 'renderings/'
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1),'pickup', 'drop', 'exit']
def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

AGENT = 'agent'
PICKUPABLE = ['hammer', 'axe', 'sticks']
BLOCKING = ['rock', 'tree']

HOLDING = 'holding'
CLIP = False
if CLIP:
    base_sprite = np.zeros((3,3,3))/10.
else:
    base_sprite = np.ones((3,3,3))/10.
sprites = []
for i in range(3):
    for j in range(3):
        new_sprite = base_sprite.copy()
        new_sprite[i,j,:] = 1.0
        sprites.append(new_sprite)

SPRITES = {'agent':sprites[0],
           'rock': sprites[1],
           'hammer': sprites[2],
           'tree': sprites[3],
           'bread': sprites[4],
           'wheat': sprites[5],
           'sticks': sprites[6],
           'axe': sprites[7],
           'house': sprites[8],
           #'wheat': sprites[9],
           }
#BGR
SPRITES['agent'][0,0] =  np.array([0/255., 0/255., 255/255.])
SPRITES['rock'][0,1] =  np.array([211/255., 211/255., 211/255.])
SPRITES['hammer'][0,2] =  np.array([204/255., 204/255., 0/255.])
SPRITES['tree'][1,0] =   np.array([34/255., 133/255., 34/255.])
SPRITES['bread'][1,1] =  np.array([0/255., 215/255., 255/255.])
SPRITES['wheat'][1,2] = np.array([10/255., 215/255., 100/255.])
SPRITES['sticks'][2,0] = np.array([45/255., 82/255., 160/255.])
SPRITES['axe'][2,1] =  np.array([255/255., 102/255., 102/255.])
SPRITES['house'][2,2] =  np.array([153/255., 52/255., 255/255.])
BIGSPRITES = copy.deepcopy(SPRITES)
SORTED_SPRITES = sorted(SPRITES.keys())
SPRITE2INDEX = {k: i for i, k in enumerate(SORTED_SPRITES)}
print("SPRITE2INDEX", SPRITE2INDEX)
ONE_HOT_SPRITES = {k : np.eye(len(SPRITES))[SPRITE2INDEX[k]] for k in SORTED_SPRITES}
print("SPRITE2INDEX", SPRITE2INDEX)
print(ONE_HOT_SPRITES)
OBJECTS =['rock', 'hammer', 'tree', 'axe', 'bread', 'sticks', 'house', 'wheat']
OBJECT_PROBS = [0.25, 0.0, 0.25, 0.0, 0.1, 0.2, 0.0, 0.2]

print("SUM OF PROB", sum(OBJECT_PROBS))

class HammerWorld(Env):
    """

    """

    metadata = {'render.modes': ['rgb', 'ansi']}

    def __init__(self, size=[10,10], res=39, add_objects=[], visible_agent=True, reward_function=None, state_obs=False, few_obj=False,
                 use_exit=False, agent_centric=True, batch_reward=False, success_function=None, goal_dim=0, pretty_renderable=False,
                 render_mode='one_hot', use_colors=False, num_colors=4
                ):
        self.nrow, self.ncol = size
        self.reward_range = (0, 1)
        self.renderres = 9
        self.agent_centric = agent_centric
        #print("hellos")
        self.ACTIONS = ACTIONS#[:-3]
        if not use_exit:
            self.ACTIONS = self.ACTIONS[:-1]
        nA = len(self.ACTIONS)
        nS = self.nrow * self.ncol
        self.add_objects = add_objects
        self.reward_function = reward_function
        self.success_function = success_function
        self.nS = nS
        self.nA = nA
        self.lastaction=None
        self.batch_reward = batch_reward
        self.visible_agent = visible_agent
        self.few_obj = few_obj
        self.episode = 0
        self.state_obs = state_obs
        self.action_space = spaces.Discrete(self.nA)
        self.goal_dim=goal_dim
        self.render_mode = render_mode
        self.use_colors = use_colors
        self.num_colors = num_colors
        if self.state_obs:
            self.render_mode = 'state'
        if self.state_obs:
            assert(self.few_obj)
            self.max_num_per_obj = 3
            self.state_space_size = len(OBJECTS)*2*self.max_num_per_obj+2+1+1
            self.observation_space = spaces.Box(low=0, high=self.nS, shape=(self.state_space_size,))
            self.state_space = self.observation_space
            self.goal_space = self.observation_space
        elif self.agent_centric:
            self.observation_space = spaces.Box(low=0, high=1., shape=((self.nrow+1)*res*2*self.ncol*res*2*3+goal_dim,))
            
        elif render_mode == 'rgb':
            self.observation_space = spaces.Box(low=0, high=1., shape=((self.nrow+1)*res*self.ncol*res*3+goal_dim,))
        elif render_mode == 'one_hot':
            res = 1
            self.observation_space = spaces.Box(low=0, high=1., shape=((self.nrow+1)*res*self.ncol*res*len(SPRITES)+goal_dim,))
            if self.use_colors:
                self.observation_space = spaces.Box(low=0, high=1., shape=((self.nrow+1)*res*self.ncol*res*(len(SPRITES)+self.num_colors)+goal_dim,))

        self.objects = []
        self.res = res

        for obj in SPRITES.keys():
            if self.render_mode == 'rgb':
                size = SPRITES[obj].shape[0]
                if size < self.res:
                    new_sprite = np.repeat(SPRITES[obj]*255,  repeats=self.res/size, axis = 1)
                    new_sprite = np.repeat(new_sprite,  repeats=self.res/size, axis =0)

                    SPRITES[obj] = new_sprite/255
                    

            size = BIGSPRITES[obj].shape[0]
            if size < self.renderres:
                new_sprite = np.repeat(BIGSPRITES[obj]*255,  repeats=self.renderres/size, axis = 1)
                new_sprite = np.repeat(new_sprite,  repeats=self.renderres/size, axis =0)
                BIGSPRITES[obj] = new_sprite/255
        
        self.BIGSPRITES= BIGSPRITES
        self.SPRITES = SPRITES

        if pretty_renderable:
            import os
            self.pretty_render_res = 30
            self.render_order = ['house', 'tree', 'rock', 'sticks', 'wheat', 'hammer', 'axe', 'bread', 'agent']
            asset_path  = '/'.join(os.path.realpath(__file__).split('/')[:-1]+['assets/*.png'])
            print("asset_path", asset_path)
            import glob
            import cv2
            asset_paths =  glob.glob(asset_path)
            
            self.pretty_render_sprites = {asset.split('/')[-1].split('.')[0]: cv2.imread(asset) for asset in asset_paths}

        

    def sample_objects(self, min_obj=None):
        num_objects = np.random.randint(15,25)
        indices = np.random.multinomial(1, OBJECT_PROBS, size=num_objects)

        indices = np.argmax(indices, axis=1)

        self.objects = []
        for obj in OBJECTS:
            i =1
            self.objects.append(obj)

            if min_obj is not None:
                while i < min_obj[obj]:
                    self.objects.append(obj)
                    i+=1
        if not self.few_obj:
            for i in range(max(num_objects-len(self.objects), 0)):
                obj_idx = indices[i]#np.random.randint(0, len(OBJECTS))
                obj = OBJECTS[obj_idx]
                self.objects.append(obj)
        return self.objects


    def from_s(self, s):
        row = int(s/self.ncol)
        return (row,s- row*self.ncol)

    def to_s(self, row, col):
        return row*self.ncol + col

    def get_root(self, obj):
        if obj is None or obj == HOLDING or obj == 'hunger' or obj == '':
            return None
        elif '_' in obj:
            return obj.split('_')[0]
        elif obj == 'agent':
            return obj

    def reset(self, init_from_state=None, min_obj=None):
        
        if init_from_state is None:
            
            self.init_from_state = False

            self.state = {}
            self.state[HOLDING] = ''
            self.state['hunger'] = 1.0
            self.state['count'] = 0
            
            if self.goal_dim > 0:
                self.goal = self.reward_function.delta_star
            self.objects = []
            self.sample_objects(min_obj=min_obj)

            self.objects += self.add_objects
            self.object_counts = {k:0 for k in OBJECTS}
            self.obj_max_index = copy.deepcopy(self.object_counts)
            self.state['obj_max_index'] = self.obj_max_index
            self.state['object_counts'] = self.object_counts
            self.state['object_positions'] = {}
            positions = np.random.permutation(self.nS)[:len(self.objects)+1]
            agent_pos =  self.from_s(positions[0])
            self.state['agent'] = agent_pos
            for i, ob in enumerate(self.objects):
                pos = self.from_s(positions[i+1])
                self.add_obj(ob, pos)
        else:
            self.init_from_state = True
            self.state = init_from_state
            self.object_counts = self.state['object_counts']
            self.obj_max_index = self.state['obj_max_index']
            self.objects = self.object_counts.keys()
        self.lastaction=None
        
        self.episode = np.random.randint(20)
        total = self.verify_env()
        self.total_count = total
        self.init_state = copy.deepcopy(self.state)
        self.episode_states = [self.init_state]
        obs = self.get_obs(mode=self.render_mode)
        self.init_img = obs.copy()
        if self.goal_dim > 0:
            obs = np.concatenate([obs.flatten(), self.goal.flatten()])
        return obs.flatten().astype(np.uint8)

    def sample_free_square(self):
        perm = np.random.permutation(self.nS)
        for s in perm:
            pos = self.from_s(s)
            if pos not in self.state.values():
                return pos

    def add_obj(self, objtype, pos):
        if objtype not in self.object_counts:
            self.object_counts[objtype] = 0
        suffix = self.obj_max_index[objtype] + 1
        self.obj_max_index[objtype] += 1
        self.object_counts[objtype] += 1
        self.state['object_positions'][objtype + '_'+str(suffix)] = pos
    def remove_obj(self, obj):
        objtype = obj.split('_')[0]
        if objtype not in self.object_counts:
            import pdb; pdb.set_trace()
        self.object_counts[objtype] -= 1
        del self.state['object_positions'][obj]

    def perform_object(self, obj):
        blocked = False
        if obj.startswith('tree'):
            if self.state[HOLDING].startswith('axe'):
                pos = self.state['object_positions'][obj]
                self.add_obj('sticks', pos)
                self.remove_obj(obj)
            else:
                blocked = True
        elif obj.startswith('rock'):
            if self.state[HOLDING].startswith('hammer'):
                 self.remove_obj(obj)
            else:
                blocked = True
        elif obj.startswith('bread') or obj.startswith('house'):
            self.state['hunger'] = 0
            if obj.startswith('bread'):
                self.remove_obj(obj)
        elif obj.startswith('sticks') and 'hammer' in self.state[HOLDING]:
            pos = self.state['object_positions'][obj]
            self.add_obj('house', pos)
            self.remove_obj(obj)
        elif obj.startswith('wheat') and 'axe' in self.state[HOLDING]:
            pos = self.state['object_positions'][obj]
            self.add_obj('bread', pos)
            self.remove_obj(obj)  
            
        return blocked

    def move_agent(self, a):
        act = ACTIONS[a]

        pos = self.state[AGENT]
        row, col = pos[0]+act[0], pos[1]+act[1]
        #Check bounds
        if row in range(self.nrow) and col in range(self.ncol):
            local_objects = []
            for obj in self.state['object_positions'].keys():
                root = self.get_root(obj)
                if root !='agent':
                    obj_pos = self.state['object_positions'][obj]
                    if obj_pos == (row, col):
                        local_objects.append(obj)
            is_blocked = False
            for obj in local_objects:
                blocked = self.perform_object(obj)
                is_blocked = blocked or is_blocked
            #Check obstacles:
            if is_blocked:
                return pos

            self.state[AGENT] = (row, col)
            if len(self.state[HOLDING]) > 0:
                obj = self.state[HOLDING]
                self.state['object_positions'][obj] = (row, col)
            return (row, col)
        else:
            return pos

    def try_pickup(self):
        pos = self.state[AGENT]
        for obj in self.state['object_positions'].keys():
            root = self.get_root(obj)
            if root is not None and root != obj:
                obj_pos = self.state['object_positions'][obj]
                if obj_pos == pos and root in PICKUPABLE:
                    if self.state[HOLDING] == '':
                        self.state[HOLDING] = obj
        return

    def try_drop(self):
        # Can only drop if nothing else is there
        pos = self.state[AGENT]
        if self.state[HOLDING] is not None:
            for obj in self.state['object_positions'].keys():
                root = self.get_root(obj)
                if root is not None and root != obj:
                    obj_pos = self.state['object_positions'][obj]
                    if obj_pos == pos and obj != self.state[HOLDING]:
                        return
            self.state[HOLDING] = ''
        return

    def verify_env(self):
        my_obj_counts = {k:0 for k in OBJECTS}
        for obj in self.state['object_positions'].keys():
            if obj != 'agent' and obj!='holding' and obj != 'object_counts' and obj!='hunger':
                objtype = obj.split('_')[0]
                if objtype not in my_obj_counts:
                    my_obj_counts[objtype] = 0
                my_obj_counts[objtype] += 1
        for k in my_obj_counts.keys():
            if my_obj_counts[k] !=self.object_counts[k]:
                import pdb; pdb.set_trace()
            assert(my_obj_counts[k] == self.object_counts[k])
        for k in self.object_counts.keys():
            assert(my_obj_counts[k] == self.object_counts[k])
        return sum(my_obj_counts.values())

    def step(self, a):
        total = self.verify_env()
        prev_state = copy.deepcopy(self.state)
        self.state['count'] +=1
        assert(total <= self.total_count)
        self.total_count = total
        r = 0
        d = False
        action = ACTIONS[a]
        if action == 'exit':
            d = True
        elif action == 'pickup':
            self.try_pickup()
        elif action == 'drop':
            self.try_drop()
        else:
            new_pos = self.move_agent(a)
        self.lastaction=a
        obs = self.get_obs(mode=self.render_mode)
        success = 0
        self.episode_states.append(copy.deepcopy(self.state))

        if self.success_function is not None:
            r = self.success_function(prev_state, self.state)
            success = self.success_function(self.init_state, self.state)>0
        #old_delta = self.goal
        if self.goal_dim > 0:
            r = self.get_reward(obs) +r

        if self.goal_dim > 0:
            obs = np.concatenate([obs.flatten(), self.goal.flatten()])
        #success = self.reward_function(self.init_state, self.state)>0
        return (obs.flatten().astype(np.uint8), r, d, {'success': success, 'count': self.state['count'], 'done':d})


    def get_obs(self, mode='rgb'):
        if self.state_obs:
            obs = self.imagine_obs(self.state)
            return obs
        if mode == 'rgb':
            img = np.zeros(((self.nrow+1)*self.res, self.ncol*self.res, 3))
            to_get_obs = self.state['object_positions'].keys()
            for obj in to_get_obs:
                root = self.get_root(obj)
                if root is not None:
                    if root in self.SPRITES:
                        row, col = self.state['object_positions'][obj]

                        img[row*self.res:(row+1)*self.res, col*self.res:(col+1)*self.res, :] += self.SPRITES[root]

            if self.visible_agent:
                row, col = self.state[AGENT]
                img[row*self.res:(row+1)*self.res, col*self.res:(col+1)*self.res, :] += self.SPRITES[AGENT]

            if self.agent_centric:
                img = self.center_agent(img, self.res)
            w,h,c = img.shape
            img[w-self.res:w, 0:self.res, 0] = self.state['hunger']
            img[w-self.res:w, self.res:self.res*2, :] = (len(self.state[HOLDING])>0)
            if CLIP:
                img = np.clip(img, 0, 1.0)
            return img.flatten()*255
        if mode == 'one_hot':
            img = np.zeros(((self.nrow+1)*self.res, self.ncol*self.res, len(SPRITES)))
            to_get_obs = self.state['object_positions'].keys()
            for obj in to_get_obs:
                root = self.get_root(obj)
                if root is not None:
                    if root in ONE_HOT_SPRITES:
                        row, col = self.state['object_positions'][obj]
                        sprite_index = SPRITE2INDEX[root]
                        
                        img[row*self.res:(row+1)*self.res, col*self.res:(col+1)*self.res, :] += ONE_HOT_SPRITES[root]

            if self.visible_agent:
                row, col = self.state[AGENT]
                img[row*self.res:(row+1)*self.res, col*self.res:(col+1)*self.res, :] += ONE_HOT_SPRITES[AGENT]

            if self.agent_centric:
                img = self.center_agent(img, self.res)
            w,h,c = img.shape
            img[w-self.res:w, 0:self.res, 0] = self.state['hunger']
            if len(self.state[HOLDING]) > 0:
                root = self.get_root(self.state[HOLDING])
                if root in ONE_HOT_SPRITES:
                    img[w-self.res:w, self.res:self.res*2, :] =  ONE_HOT_SPRITES[root]
            if CLIP:
                img = np.clip(img, 0, 1.0)
            return img.flatten()
        if mode == 'ascii':
            img = np.zeros((self.nrow, self.ncol)).astype(np.str)
            for obj in self.state.keys():
                root = self.get_root(obj)
                if root is not None:
                    if  '_' in obj:
                        objtype, obj_num  = obj.split('_')
                        row, col = self.state[obj]
                        img[row, col] = objtype[0] + obj_num[0]
                    elif 'agent' in obj:
                        row, col = self.state[obj]
                        img[row, col] = 'AG'
            img[img=='0.0'] = '  '
            return img
        
    def imagine_obs(self, state, mode='rgb'):
        if self.state_obs:
            obs = np.zeros(self.state_space_size)
            obs[:2] = state['agent']
            for obj, pos in state['object_positions'].items():
                root, num = obj.split('_')
                num = int(num)-1
                assert(num <3)
                assert(num >=0)
                idx = OBJECTS.index(root)*2*self.max_num_per_obj+2+num*2
                obs[idx:idx+2] = pos
            obs[-2] = state['hunger']
            obs[-1] =  (len(state[HOLDING])>0)
                
            return obs
        if mode == 'rgb':
            img = np.zeros(((self.nrow+1)*self.res, self.ncol*self.res, 3))
            to_get_obs = state['object_positions'].keys()
            for obj in to_get_obs:
                root = self.get_root(obj)
                if root is not None:
                    if root in self.SPRITES:
                        row, col = state['object_positions'][obj]

                        img[row*self.res:(row+1)*self.res, col*self.res:(col+1)*self.res, :] += self.SPRITES[root]

            if self.visible_agent:
                row, col = state[AGENT]
                img[row*self.res:(row+1)*self.res, col*self.res:(col+1)*self.res, :] += self.SPRITES[AGENT]

            
            if self.agent_centric:
                img = self.center_agent(img, self.res)
            w,h,c = img.shape
            img[w-self.res:w, 0:self.res, 0] = self.state['hunger']
            img[w-self.res:w, self.res:self.res*2, :] = (len(self.state[HOLDING])>0)
            return img.flatten()

    def center_agent(self, img, res):
        new_obs = np.zeros((img.shape[0]*2, img.shape[1]*2, 3))+0.1
        row, col = self.state[AGENT]
        disp_x = img.shape[0] - row*res
        disp_y = img.shape[1] - col*res
        new_obs[disp_x:disp_x+img.shape[0], disp_y:disp_y + img.shape[1]] = img
        return new_obs

    def render(self, mode='rgb'):
        import cv2

        if mode == 'rgb':
            img = np.zeros(((self.nrow+1)*self.renderres, self.ncol*self.renderres, 3))
            to_get_obs = self.state['object_positions'].keys()
            for obj in to_get_obs:
                root = self.get_root(obj)
                if root is not None:
                    if root in self.SPRITES:
                        row, col = self.state['object_positions'][obj]

                        img[row*self.renderres:(row+1)*self.renderres, col*self.renderres:(col+1)*self.renderres, :] += self.BIGSPRITES[root]

            if self.visible_agent:
                row, col = self.state[AGENT]
                img[row*self.renderres:(row+1)*self.renderres, col*self.renderres:(col+1)*self.renderres, :] += self.BIGSPRITES[AGENT]

            if self.agent_centric:
                img = self.center_agent(img, self.renderres)
            w,h,c = img.shape
            img[w-self.renderres:w, 0:self.renderres, 0] = self.state['hunger']
            img[w-self.renderres:w, self.renderres:self.renderres*2, :] = (len(self.state[HOLDING])>0)
            cv2.imwrite(RENDER_DIR+'img{:04d}_{:04d}.png'.format(self.episode, self.state['count']), img*255)
            
    def pretty_render(self, mode='rgb'):
        import cv2
        if mode == 'rgb':
            img = np.zeros(((self.nrow+1)*self.pretty_render_res, self.ncol*self.pretty_render_res, 3)).astype(np.uint8)
            grass = (self.pretty_render_sprites['grass2']/3).astype(np.uint8)
            for row in range(self.nrow):
                for col in range(self.ncol):
                    img[row*self.pretty_render_res:(row+1)*self.pretty_render_res, col*self.pretty_render_res:(col+1)*self.pretty_render_res] = grass
            to_get_obs = self.state['object_positions'].keys()
            for to_render_obj in self.render_order:
                if to_render_obj == 'agent':
                    sprite = self.pretty_render_sprites[to_render_obj]
                    row, col = self.state[AGENT]
                    gray_pixels = np.max(sprite, axis=2)
                    idx = np.where(gray_pixels > 0)
                    col_offset = col*self.pretty_render_res
                    row_offset = row*self.pretty_render_res
                    img[(idx[0]+row_offset, idx[1]+col_offset)] = sprite[idx]
                else:
                    for obj in to_get_obs:
                        root = self.get_root(obj)
                        if root == to_render_obj:
                            #This code is to layer the sprites properly: Don't blend the colors
                            sprite = self.pretty_render_sprites[to_render_obj]
                            row, col = self.state['object_positions'][obj]
                            gray_pixels = np.max(sprite, axis=2)
                            idx = np.where(gray_pixels > 0)
                            col_offset = col*self.pretty_render_res
                            row_offset = row*self.pretty_render_res
                            img[(idx[0]+row_offset, idx[1]+col_offset)] = sprite[idx]


            if self.agent_centric:
                img = self.center_agent(img, self.pretty_render_res)
            w,h,c = img.shape

            if len(self.state[HOLDING])>0:
                root = self.get_root(self.state[HOLDING])
                img[w-self.pretty_render_res:w, self.pretty_render_res:self.pretty_render_res*2] = self.pretty_render_sprites[root]
            cv2.imwrite(RENDER_DIR+'pretty_img{:04d}_{:04d}.png'.format(self.episode, self.state['count']), img)
            return img

    def check_move_agent(self, a):
        """ Does not actually change state"""
        act = ACTIONS[a]

        pos = self.state[AGENT]
        row, col = pos[0]+act[0], pos[1]+act[1]
        #Check bounds
        removes_obj = None
        blocked = False
        if row in range(self.nrow) and col in range(self.ncol):
            local_objects = []
            for obj in self.state['object_positions'].keys():
                root = self.get_root(obj)
                if root !='agent':
                    obj_pos = self.state['object_positions'][obj]
                    if obj_pos == (row, col):
                        local_objects.append(obj)
            is_blocked = False
            for obj in local_objects:
                blocked = False
                if obj.startswith('tree'):
                    if not self.state[HOLDING].startswith('axe'):
                        blocked = True
                    else:
                        removes_obj = 'tree'
                elif obj.startswith('rock'):
                    if not self.state[HOLDING].startswith('hammer'):
                        blocked = True
                    else:
                        removes_obj = 'rock'
                elif obj.startswith('bread'):
                    removes_obj= 'bread'
                elif obj.startswith('wheat') and self.state[HOLDING].startswith('axe'):
                    removes_obj = 'wheat'
        else:
            blocked = True
        if blocked:
            row, col = pos
        return (row, col), blocked, removes_obj

    def get_diagnostics(self,paths, **kwargs):
        successes = [p['env_infos'][-1]['success'] for p in paths]
        success_rate = sum(successes)/len(successes)
        lengths = [p['env_infos'][-1]['count'] for p in paths]
        length_rate = sum(lengths)/len(lengths)
        return {'SuccessRate': success_rate, 'PathLengthMean': length_rate, 'PathLengthMin':min(lengths)}
