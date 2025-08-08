import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from pufferlib import PufferEnv
from gymnasium.spaces.utils import flatdim
from gym.spaces import Box, Discrete
from pufferlib.environments.diayn_wrapper import env_creator
import importlib
from pufferlib.pufferl import load_config
from pufferlib.ocean.environment import MAKE_FUNCTIONS
from pufferlib import pytorch

#/Users/ha24583/Documents/GitHub/PufferLib/experiments/puffer_stacking_sunbwh25.pt
import importlib

def resolve_class_from_instance(obj):
    module = importlib.import_module(type(obj).__module__)
    return getattr(module, type(obj).__name__)
import inspect
import importlib

def clone_env(base_env):
    # Step 1: resolve the class
    module = importlib.import_module(type(base_env).__module__)
    cls = getattr(module, type(base_env).__name__)

    # Step 2: inspect constructor
    sig = inspect.signature(cls.__init__)

    # Step 3: build kwargs
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if hasattr(base_env, name):
            kwargs[name] = getattr(base_env, name)
        elif param.default is not inspect._empty:
            kwargs[name] = param.default
        else:
            raise ValueError(f"Cannot infer value for required arg: '{name}'")

    # Step 4: make new instance
    return cls(**kwargs)

def find_key_from_value(d, item):
    return next((k for k, v in d.items() if v is item), None)


class DIAYNMetaEnv(PufferEnv):
    """
    A drop-in VecEnv wrapper that replaces the
    rewards produced by the underlying C backend.
    """
    def __init__(self, base_env, diayn_model= '/Users/ha24583/Documents/GitHub/PufferLib/experiments/puffer_stacking_sunbwh25.pt', k =2, skill_sampler=None, num_skills=4, disc_train_interval=5, disc_batch_size=1024, device="cpu"):
        assert diayn_model is not None, "Diayn model must be provided"
        self.diayn_model_dict = torch.load(diayn_model, map_location=device)
        diayn_model_obs_shape = self.diayn_model_dict['policy.encoder.0.weight'].shape[1]
        self.num_skills = num_skills
        dummy_env = clone_env(base_env)
       
        #Re-shape the dummy_env obs space to add skill 
        old_low  = dummy_env.single_observation_space.low
        old_high = dummy_env.single_observation_space.high
        
        old_low_flat  = old_low .astype(np.float32).flatten()   
        old_high_flat = old_high.astype(np.float32).flatten()

        low  = np.concatenate([old_low_flat,  np.zeros(self.num_skills, dtype=np.float32)])
        high = np.concatenate([old_high_flat, np.ones (self.num_skills, dtype=np.float32)])  
        dummy_env.single_observation_space = Box(low=low, high=high, dtype=np.float32) 
        env_name = 'puffer_' + find_key_from_value(MAKE_FUNCTIONS, type(base_env).__name__)
        args = load_config(env_name)
        module_name = 'pufferlib.ocean' #TODO: this should be automated
        env_module =  importlib.import_module(module_name)
        policy_cls = getattr(env_module.torch, args['policy_name']) 
        policy = policy_cls(dummy_env, **args['policy'])
        rnn_name = args['rnn_name']
        if rnn_name is not None:
            rnn_cls = getattr(env_module.torch, args['rnn_name'])
            policy = rnn_cls(dummy_env, policy, **args['rnn'])
        policy = policy.to(device)
        policy.load_state_dict(self.diayn_model_dict)
        self.diayn_model = policy
        dummy_env.close()
        del dummy_env

        self.diayn_model.eval()
        self.env = base_env
        self.num_skills = num_skills
        self.device = device
        self.k = k

        self.single_action_space = Discrete(self.num_skills)

        self.step_count = 0
                # LSTM
        if policy.lstm:
            n = self.env.num_agents #vecenv.agents_per_batch   #self.agents_per_batch = self.driver_env.num_agents * num_envs

            total_agents = self.env.num_agents #vecenv.num_agents #self.num_agents= self.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
            self.lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}

      
    @property
    def single_observation_space(self):
        return self.single_observation_and_skill_space

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def _skill_obs(self, obs,actions):
        """
        Flattens the observations and concatenates the one-hot skill vector
        """
        flat = obs.reshape(obs.shape[0], -1)
        one_hot = np.eye(self.num_skills, dtype=np.float32)[actions]
        return np.concatenate([flat, one_hot], axis=1)


    def reset(self, *a, **kw):
        self.step_count = 0
        return self.env.reset(*a, **kw)      

    def close(self):
        self.env.close()

    def step(self, actions):
        
        with torch.no_grad():

            for i in range(self.k):
                state = dict()
                state['lstm_h'] = self.lstm_h[0]
                state['lstm_c'] = self.lstm_c[0]
                current_obs = self.env.observations
                skill_obs = self._skill_obs(current_obs, actions)
                skill_obs = torch.as_tensor(skill_obs).to(self.device)
                logits, values = self.diayn_model.forward_eval(skill_obs, state)
                action, logprob, _ = pytorch.sample_logits(logits)
                obs, rew, term, trunc, info = self.env.step(action)


        if not info:
            info.append({})

        log_dict = info[0]

        self.step_count += 1

        return obs, rew, term, trunc, info






