## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import os
import sys
import glob
import ast
import time
import random
import shutil
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
rich.traceback.install(show_locals=False)

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

# Assume advantage kernel has been built if CUDA compiler is available
ADVANTAGE_CUDA = shutil.which("nvcc") is not None

class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = config['torch_deterministic']
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config['seed']
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        self.is_continuous = isinstance(atn_space,
                pufferlib.spaces.Box)

        

        
        buffer_size = int(float(config['buffer_size']))

        segments = int(config['buffer_size'] // total_agents)
        segments = max(segments, 1)
        learning_starts = int(0.05 * config['buffer_size'])  # 5 % of total transitions
        if buffer_size < config['learning_starts']:
            raise pufferlib.APIUsageError(
                f'buffer_size {buffer_size} must be >= learning_starts {config["learning_starts"]}'
            )
        self.segments = segments
        device = config['device']
        self.observations = torch.zeros(segments, total_agents, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self.next_observations = torch.zeros(segments, total_agents, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self.actions = torch.zeros(segments, total_agents, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.values = torch.zeros(segments, total_agents, device=device)
        self.logprobs = torch.zeros(segments, total_agents, device=device)
        self.rewards = torch.zeros(segments, total_agents, device=device)
        self.terminals = torch.zeros(segments, total_agents, device=device)
        self.truncations = torch.zeros(segments, total_agents, device=device)

        self.pos = 0
        self.buffer_full = False

        #buffers to hold obs and actions until next step so it can align in the buffers above 

        self.obs_buf = torch.zeros(total_agents, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and config['cpu_offload'],
            device='cpu' if config['cpu_offload'] else device)
        self.act_buf = torch.zeros(total_agents, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.has_state = torch.zeros(total_agents, device=device, dtype=torch.bool)
        # Minibatching & gradient accumulation
        minibatch_size = config['minibatch_size']

        if buffer_size < minibatch_size:
            raise pufferlib.APIUsageError(
                f'batch_size {buffer_size} must be >= minibatch_size {minibatch_size}'
            )
        self.minibatch_size = minibatch_size

        # Torch compile
        #self.uncompiled_policy = policy
        #self.policy = policy
        #if config['compile']:
         #   self.policy = torch.compile(policy, mode=config['compile_mode'])
          #  self.policy.forward_eval = torch.compile(policy, mode=config['compile_mode'])
           # pufferlib.pytorch.sample_logits = torch.compile(pufferlib.pytorch.sample_logits, mode=config['compile_mode'])

          

        from pufferlib.models import SoftQNetwork, Actor
        hidden_size = config['hidden_size']
        self.actor = Actor(vecenv.driver_env, hidden_size).to(device)
        self.qf1 = SoftQNetwork(vecenv.driver_env, hidden_size).to(device)
        self.qf2 = SoftQNetwork(vecenv.driver_env, hidden_size).to(device)
        self.qf1_target = SoftQNetwork(vecenv.driver_env, hidden_size).to(device)
        self.qf2_target = SoftQNetwork(vecenv.driver_env, hidden_size).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=config['q_lr'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['policy_lr'])
        self.learning_starts = learning_starts
        self.target_network_update_freq = config['target_network_update_freq']
        self.policy_freq = config['policy_freq']
        self.tau = config['tau']
        self.warmup = True
        
        #DOES NOT HANDLE MULTD yet? 
        if config['autotune']:

            self.autotune = True
            if self.is_continuous:
                self.target_entropy = -torch.prod(torch.Tensor(atn_space.shape).to(device)).item()

            else:
                n_act = atn_space.n
                self.target_entropy_scale = config['target_entropy_scale']
                self.target_entropy = - self.target_entropy_scale *(torch.log(1/torch.tensor(float(n_act), device=device)))

            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=config['q_lr'])
        else:
            self.alpha = config['alpha']
            self.autotune = False

        self.uncompiled_actor = self.actor

        if config['compile']:
            self.actor = torch.compile(self.actor, mode=config['compile_mode'])
            self.actor.get_action = torch.compile(self.actor.get_action, mode=config['compile_mode'])
            self.actor.get_action_eval = torch.compile(self.actor.get_action_eval, mode=config['compile_mode'])

            self.qf1 = torch.compile(self.qf1, mode=config['compile_mode'])
            self.qf2 = torch.compile(self.qf2, mode=config['compile_mode'])
            self.qf1_target = torch.compile(self.qf1_target, mode=config['compile_mode'])
            self.qf2_target = torch.compile(self.qf2_target, mode=config['compile_mode'])
     


       
        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)


        # Automatic mixed precision
        precision = config['precision']
        self.amp_context = contextlib.nullcontext()
        if config.get('amp', True) and config['device'] == 'cuda':
            self.amp_context = torch.amp.autocast(device_type='cuda', dtype=getattr(torch, precision))
        if precision not in ('float32', 'bfloat16'):
            raise pufferlib.APIUsageError(f'Invalid precision: {precision}: use float32 or bfloat16')

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}

        self.last_log_epoch = 0
        self.start_time = time.time()
        self.runtime = 0

        # Dashboard
        models = [self.actor, self.qf1, self.qf2, self.qf1_target, self.qf2_target]
        self.model_size = sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    @property
    def eps(self):
        if self.epoch == self.last_log_epoch:
            return 0
        rate = (self.epoch - self.last_log_epoch) / (time.time() - self.last_log_time)
        return round(rate, 1)

    
    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']
        env_id = slice(0, 1)


        while self.warmup or env_id.stop < self.total_agents:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t # TODO: Handle truncations separately
            self.global_step += int(mask.sum())

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)#, non_blocking=True)
            r = torch.as_tensor(r).to(device)#, non_blocking=True)
            d = torch.as_tensor(d).to(device)#, non_blocking=True)

            t = torch.as_tensor(t).to(device)#, non_blocking=True)


            #TODO NEXT OBS HANDLING IN TRUNCTIONS. AND TERMS?

            profile('eval_forward', epoch)
            with torch.no_grad(), self.amp_context:

                if self.global_step <= self.learning_starts:
                    action = np.array([self.vecenv.single_action_space.sample() for _ in range(env_id.stop-env_id.start)])
                else:
                    action = self.actor.get_action_eval(o_device)
                    action.detach()

            profile('eval_copy', epoch)
            with torch.no_grad():

                if self.has_state[env_id].any():

                    #MAUY NEED TO HANDLE REAL NEXT OBS FROM TRUNCTIONS


                    if config['cpu_offload']:
                        self.observations[self.pos, env_id] = self.obs_buf[env_id]
                        self.next_observations[self.pos, env_id] = o
                    else:
                        self.observations[self.pos, env_id] = self.obs_buf[env_id].to(device)
                        self.next_observations[self.pos, env_id] = o_device

                    done_mask = (d.bool() | t.bool())  #To handle truncs so we dont false bootstrap

                    self.actions[self.pos, env_id] = self.act_buf[env_id]
                    self.rewards[self.pos, env_id] = r
                    self.terminals[self.pos, env_id] = done_mask.float()
                    self.truncations[self.pos, env_id] = t

                    if env_id.stop == self.total_agents:
                        self.pos += 1

             
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                

            profile('eval_misc', epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)
            with torch.no_grad():

                if config['cpu_offload']:
                    self.obs_buf[env_id] = o
                else:
                    self.obs_buf[env_id] = o_device
                self.has_state[env_id] = True 
                self.act_buf[env_id] = torch.as_tensor(action, device=device)

            profile('env', epoch)
            self.vecenv.send(action)

            if self.pos*self.total_agents >= self.learning_starts:
                self.warmup = False

        if self.pos + 1 >= self.segments:
            self.pos = 0
            self.buffer_full = True

        env_id = slice(0, 1)


        profile('eval_misc', epoch)
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        losses = defaultdict(float)
        config = self.config
        device = config['device']
        self.gamma = config['gamma']

        updates = max(1, int(self.total_agents * config['train_ratio']))

        for _ in range(updates):
            profile('train', epoch)



            profile('train_misc', epoch, nest=True)
            self.amp_context.__enter__()

            profile('train_copy', epoch)
         
            
            if self.buffer_full:
                row_idx = torch.randint(0, self.segments, (self.minibatch_size,), device=device)
            else:
                row_idx = torch.randint(1, self.pos, (self.minibatch_size,),device=device)
            agent_idx = torch.randint(0, self.total_agents, (self.minibatch_size,),device=device)

            mb_obs = self.observations[row_idx, agent_idx]
            mb_next_obs = self.next_observations[row_idx, agent_idx]
            mb_actions = self.actions[row_idx, agent_idx]
            mb_rewards = self.rewards[row_idx, agent_idx].view(-1,1)
            mb_terminals = self.terminals[row_idx, agent_idx].view(-1,1)
            mb_truncations = self.truncations[row_idx, agent_idx].view(-1,1)
           
           
            
            
            with torch.no_grad():

                profile('train_actor_forward', epoch)
                next_state_actions, next_state_logpi, next_state_action_probs = self.actor.get_action(mb_next_obs)

                profile('train_q_forward', epoch)
                if self.is_continuous:
                    qf1_next_target = self.qf1_target(mb_next_obs, next_state_actions)
                    qf2_next_target = self.qf2_target(mb_next_obs, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_logpi
                    next_q_value = mb_rewards + (1 - mb_terminals) * self.gamma * min_qf_next_target

                else:
                    qf1_next_target = self.qf1_target(mb_next_obs)
                    qf2_next_target = self.qf2_target(mb_next_obs)
                    min_qf_next_target = next_state_action_probs*(torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_logpi)
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = mb_rewards.squeeze(1) + (1 - mb_terminals.squeeze(1)) * self.gamma * min_qf_next_target




            if self.is_continuous:
                qf1_a_values = self.qf1(mb_obs, mb_actions)
                qf2_a_values = self.qf2(mb_obs, mb_actions)
            else:
                qf1_values = self.qf1(mb_obs)
                qf2_values = self.qf2(mb_obs)
                qf1_a_values = qf1_values.gather(1, mb_actions.long().unsqueeze(1)).view(-1)  
                qf2_a_values = qf2_values.gather(1, mb_actions.long().unsqueeze(1)).view(-1)


            qf1_loss = torch.nn.functional.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = torch.nn.functional.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

                             
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if self.epoch % self.policy_freq == 0:
                for _ in range(self.policy_freq):

                    profile('train_pol_actor_forward', epoch)
                    pi, log_pi,action_probs = self.actor.get_action(mb_obs) #actoin probs for discrete, pi is action

                    profile('train_pol_q_forward', epoch)
                    if self.is_continuous:
                        qf1_pi = self.qf1(mb_obs, pi)
                        qf2_pi = self.qf2(mb_obs, pi)
                    else:
                        qf1_pi = self.qf1(mb_obs)
                        qf2_pi = self.qf2(mb_obs)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    if self.is_continuous:
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                    else:
                        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_pi)).sum(dim=1).mean()



                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    losses['actor_loss'] = actor_loss.item()

                    if self.autotune:
                        with torch.no_grad():
                            profile('train_tune_actor_forward', epoch)
                            _, log_pi, action_probs = self.actor.get_action(mb_obs)
                        if self.is_continuous:
                            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                        else:
                            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).sum(dim=1).mean()
                         
                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()



                        losses['alpha_loss'] = alpha_loss.item()
                        losses['alpha'] = self.alpha
                                


            if self.epoch % self.target_network_update_freq == 0:
                profile('train_network_update_copy', epoch)
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            profile('train_misc', epoch)
        
            # Logging
            profile('train_misc', epoch)
            losses['qf1_values'] = qf1_a_values.mean().item()
            losses['qf2_values'] = qf2_a_values.mean().item()
            losses['qf1_loss'] = qf1_loss.item()
            losses['qf2_loss'] = qf2_loss.item()
            losses['qf_loss'] = qf_loss.item()

            profile.end()
            logs = None
            self.epoch += 1

        curr_time = time.time()
        self.runtime = curr_time - self.start_time

        done_training = self.global_step >= config['total_timesteps'] or self.runtime > config['max_runtime']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            self.last_log_epoch = self.epoch
            profile.clear()

        if self.global_step % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.global_step}'

        return logs
    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config['device']
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            'SPS': dist_sum(self.sps, device),
            'EPS': dist_sum(self.eps, device),
            'agent_steps': agent_steps,
            'uptime': time.time() - self.start_time,
            'epoch': int(dist_sum(self.epoch, device)),
            **{f'environment/{k}': v for k, v in self.stats.items()},
            **{f'losses/{k}': v for k, v in self.losses.items()},
            **{f'performance/{k}': v['elapsed'] for k, v in self.profile},
            #**{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            #**{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            #**{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               self.logger.log(logs, agent_steps)
               return logs
           else:
               return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}.pt')
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        run_id = self.logger.run_id
        path = os.path.join(self.config['data_dir'], f'{self.config["env"]}_{run_id}')
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f'model_{self.config["env"]}_{self.epoch:06d}.pt'
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_actor.state_dict(), model_path)

        state = {
            'global_step': self.global_step,
            'agent_step': self.global_step,
            'update': self.epoch,
            'model_name': model_name,
            'run_id': run_id,
        }
        state_path = os.path.join(path, 'trainer_state.pt')
        torch.save(state, state_path + '.tmp')
        os.replace(state_path + '.tmp', state_path)
        return model_path

    def print_dashboard(self, clear=False, idx=[0],
            c1='[cyan]', c2='[dim default]', b1='[bright_cyan]', b2='[default]'):
        config = self.config
        sps = dist_sum(self.sps, config['device'])
        eps = dist_sum(self.eps, config['device'])
        agent_steps = dist_sum(self.global_step, config['device'])
        if torch.distributed.is_initialized():
           if torch.distributed.get_rank() != 0:
               return
 
        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True,
            show_header=False, border_style='bright_cyan')
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f'{b1}PufferLib {b2}3.0 {idx[0]*" "}:blowfish:',
            f'{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%',
            f'{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%',
            f'{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%',
            f'{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%',
        )
        idx[0] = (idx[0] - 1) % 10
            
        s = Table(box=None, expand=True)
        remaining = f'{b2}A hair past a freckle{c2}'
        if sps != 0:
            remaining = duration((config['total_timesteps'] - agent_steps)/sps, b2, c2)

        s.add_column(f"{c1}Summary", justify='left', vertical='top', width=10)
        s.add_column(f"{c1}Value", justify='right', vertical='top', width=14)
        s.add_row(f'{b2}Env', f'{b2}{config["env"]}')
        s.add_row(f'{b2}Params', abbreviate(self.model_size, b2, c2))
        s.add_row(f'{b2}Steps', abbreviate(agent_steps, b2, c2))
        s.add_row(f'{b2}SPS', abbreviate(sps, b2, c2))
        s.add_row(f'{b2}EPS', abbreviate(eps, b2, c2))
        s.add_row(f'{b2}Gradient_steps', f'{b2}{self.epoch}')
        s.add_row(f'{b2}Epoch', f'{b2}{self.epoch}')
        s.add_row(f'{b2}Uptime', duration(self.uptime, b2, c2))
        s.add_row(f'{b2}Remaining', remaining)

        delta = profile.eval['buffer'] + profile.train['buffer']
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf('Evaluate', b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf('  Env', b2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf('Train', b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf('  Act_Forward', c2, delta, profile.train_actor_forward, b2, c2))
        p.add_row(*fmt_perf('  Q_Forward', c2, delta, profile.train_q_forward, b2, c2))
        p.add_row(*fmt_perf('  Pol_Act_Forward', c2, delta, profile.train_pol_actor_forward, b2, c2))
        p.add_row(*fmt_perf('  Pol_Q_Forward', c2, delta, profile.train_pol_q_forward, b2, c2))
        p.add_row(*fmt_perf('  Tune_Act_Forward', c2, delta, profile.train_tune_actor_forward, b2, c2))
        p.add_row(*fmt_perf('  Forward', b2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf('  Learn', b2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf('  Copy', b2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf('  Misc', b2, delta, profile.train_misc, b2, c2))

        l = Table(box=None, expand=True, )
        l.add_column(f'{c1}Losses', justify="left", width=16)
        l.add_column(f'{c1}Value', justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            try: # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f'{b2}{metric}', f'{b2}{value:.3f}')
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print('\033[0;0H' + capture.get())

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    '''CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version.'''

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return f'{b2}{num}{c2}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}K'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}M'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}B'
    else:
        return f'{b2}{num/1e12:.2f}{c2}T'

def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100*prof['buffer']/delta_ref - 1e-5)
    return f'{color}{name}', duration(prof['elapsed'], b2, c2), f'{b2}{percent:2d}{c2}%'

def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()

def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()

class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]['start'] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile['start']
        profile['delta'] += delta
        # Multiply delta by freq to account for skipped epochs
        profile['elapsed'] += delta * self.frequency

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof['delta'] > 0:
                prof['buffer'] = prof['delta']
                prof['delta'] = 0

class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100*psutil.cpu_percent()/psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100*mem.active/mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                   time.sleep(self.delay)
                   continue

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100*(total-free)/total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]

class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100*time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path):
        pass

class NeptuneLogger:
    def __init__(self, args, load_id=None, mode='async'):
        import neptune as nept
        neptune_name = args['neptune_name']
        neptune_project = args['neptune_project']
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune['model'].track_files(model_path)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination='artifacts')
        return f'artifacts/{self.run_id}.pt'
 
class WandbLogger:
    def __init__(self, args, load_id=None, resume='allow'):
        import wandb
        wandb.init(
            id=load_id or wandb.util.generate_id(),
            project=args['wandb_project'],
            group=args['wandb_group'],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags = [args['tag']] if args['tag'] is not None else [],
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args['no_model_upload']

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type='model')
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f'{self.run_id}:latest')
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f'{data_dir}/{model_file}'

def train(env_name, args=None, vecenv=None, policy=None, logger=None, should_stop_early=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print("World size", world_size)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if 'LOCAL_RANK' in os.environ:
        args['train']['device'] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend='nccl', world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank
        )
        if hasattr(policy, 'lstm'):
            #model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args['neptune']:
        logger = NeptuneLogger(args)
    elif args['wandb']:
        logger = WandbLogger(args)

    train_config = { **args['train'], 'env': env_name }
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    all_logs = []
    while pufferl.global_step < train_config['total_timesteps'] and pufferl.runtime < args['train']['max_runtime']:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            if pufferl.global_step > 0.20*train_config['total_timesteps']:
                all_logs.append(logs)

            if should_stop_early is not None and should_stop_early(logs):
                model_path = pufferl.close()
                pufferl.logger.close(model_path)
                return all_logs

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    for i in range(128):  # Run eval for at least 32, but put a hard stop at 128.
        stats = pufferl.evaluate()
        if i >= 32 and stats:
            break

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
    return all_logs

def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    backend = args['vec']['backend']
    if backend != 'PufferEnv':
        backend = 'Serial'

    args['vec'] = dict(backend=backend, num_envs=1)
    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]
    device = args['train']['device']

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    frames = []
    while True:
        render = driver.render()
        if len(frames) < args['save_frames']:
            frames.append(render)

        # Screenshot Ocean envs with F12, gifs with control + F12
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1/args['fps'])
        elif driver.render_mode == 'rgb_array':
            pass
            #import cv2
            #render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #cv2.imshow('frame', render)
            #cv2.waitKey(1)
            #time.sleep(1/args['fps'])

        with torch.no_grad():
            ob = torch.as_tensor(ob).to(device)
            logits, value = policy.forward_eval(ob, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob = vecenv.step(action)[0]

        if len(frames) > 0 and len(frames) == args['save_frames']:
            import imageio
            imageio.mimsave(args['gif_path'], frames, fps=args['fps'], loop=0)
            print(f'Saved {len(frames)} frames to {args["gif_path"]}')

def stop_if_loss_nan(logs):
    return any("losses/" in k and np.isnan(v) for k, v in logs.items())

def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args['wandb'] and not args['neptune']:
        raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')

    method = args['sweep'].pop('method')
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'
    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        sweep.suggest(args)
        all_logs = train(env_name, args=args, should_stop_early=stop_if_loss_nan)
        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = downsample([log['agent_steps'] for log in all_logs], points_per_run)

        if len(timesteps) > 0 and timesteps[-1] < 0.7 * total_timesteps:  # 0.7 is arbitrary
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps

def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args['train'], env=args['env_name'], tag=args['tag'])
    pufferl = PuffeRL(train_config, vecenv, policy, neptune=args['neptune'], wandb=args['wandb'])

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    prof.export_chrome_trace("trace.json")

def export(args=None, env_name=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    args['vec'] = dict(backend='Serial', num_envs=1)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    path = f'{args["env_name"]}_weights.bin'
    weights = np.concatenate(weights)
    weights.tofile(path)
    print(f'Saved {len(weights)} weights to {path}')

def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    env_name = args['env_name']
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
 
def load_env(env_name, args):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args['env'], **args['vec'])

def load_policy(args, vecenv, env_name=''):
    package = args['package']
    module_name = 'pufferlib.ocean' if package == 'ocean' else f'pufferlib.environments.{package}'
    env_module = importlib.import_module(module_name)

    device = args['train']['device']
    policy_cls = getattr(env_module.torch, args['policy_name'])
    policy = policy_cls(vecenv.driver_env, **args['policy'])

    rnn_name = args['rnn_name']
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])
        policy = rnn_cls(vecenv.driver_env, policy, **args['rnn'])

    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['neptune']:
            path = NeptuneLogger(args, load_id, mode='read-only').download()
        elif args['wandb']:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError('No run id provided for eval')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        #state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        #optim_state = torch.load(state_path)['optimizer_state_dict']
        #pufferl.optimizer.load_state_dict(optim_state)

    return policy

def load_config(env_name, parser=None):
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')
    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split(): break
        else:
            raise pufferlib.APIUsageError('No config for env_name {}'.format(env_name))

    return process_config(p, parser=parser)

def load_config_file(file_path, fill_in_default=True, parser=None):
    if not os.path.exists(file_path):
        raise pufferlib.APIUsageError('No config file found')

    config_paths = [file_path]

    if fill_in_default:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
        # Process the puffer defaults first
        config_paths.insert(0, os.path.join(puffer_dir, 'config/default.ini'))

    p = configparser.ConfigParser()
    p.read(config_paths)

    return process_config(p, parser=parser)

def make_parser():
    '''Creates the argument parser with default PufferLib arguments.'''
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--load-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--load-id', type=str,
        default=None, help='Kickstart/eval from from a finished Wandb/Neptune run')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='eval.gif')
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true', help='Use neptune for logging')
    parser.add_argument('--neptune-name', type=str, default='pufferai')
    parser.add_argument('--neptune-project', type=str, default='ablations')
    parser.add_argument('--no-model-upload', action='store_true', help='Do not upload models to wandb or neptune')
    parser.add_argument('--local-rank', type=int, default=0, help='Used by torchrun for DDP')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    return parser

def process_config(config, parser=None):
    if parser is None:
        parser = make_parser()

    parser.description = f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]' \
        ' demo options. Shows valid args for your env and policy'

    def auto_type(value):
        """Type inference for numeric args that use 'auto' as a default value"""
        if value == 'auto': return value
        if value.isnumeric(): return int(value)
        return float(value)

    for section in config.sections():
        for key in config[section]:
            try:
                value = ast.literal_eval(config[section][key])
            except:
                value = config[section][key]

            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=value,
                type=auto_type if value == 'auto' else type(value)
            )

    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args['train']['env'] = args['env_name'] or ''  # for trainer dashboard
    args['train']['use_rnn'] = args['rnn_name'] is not None
    return args

def main():
    err = 'Usage: puffer [train, eval, sweep, autotune, profile, export] [env_name] [optional args]. --help for more info'
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == 'train':
        train(env_name=env_name)
    elif mode == 'eval':
        eval(env_name=env_name)
    elif mode == 'sweep':
        sweep(env_name=env_name)
    elif mode == 'autotune':
        autotune(env_name=env_name)
    elif mode == 'profile':
        profile(env_name=env_name)
    elif mode == 'export':
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)

if __name__ == '__main__':
    main()
