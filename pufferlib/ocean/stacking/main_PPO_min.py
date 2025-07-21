import os
from stable_baselines3 import PPO
from stacking_paper_env import ContainerStackEnv
from stable_baselines3.common.monitor import Monitor
import random 
import torch as th
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

def generate_training_set(scale, num_instances=100):
    """
    Generate 'num_instances' container lists, each of length = scale.
    Returns a list of lists. E.g.:
      [
        [2, 0, 1, 4, 3],   # instance 1
        [1, 0, 2, 3, 4],   # instance 2
        ...
      ]
    """
    instances = []
    for _ in range(num_instances):
        # Example: shuffle a list of priorities from 0..scale-1
        containers = list(range(scale))
        random.shuffle(containers)
        instances.append(containers)
    return instances

def mask_fn(env: ContainerStackEnv) -> np.ndarray:
    return env.get_action_mask()

def train_for_scale(scale, max_height=3, max_steps=100, total_timesteps=20000, model_save_path='models'):
    """
    Train a PPO model on a single scale, using random container instances.
    We'll generate a new instance each time we reset by re-creating the env. 
    
    :param scale: number of containers
    :param max_height: stack height
    :param max_steps: optional time limit
    :param total_timesteps: how many steps to train
    :param model_save_path: directory to save the model
    :return: the trained model
    """
    # Create a log directory
    log_dir = f"logs/scale_{scale}"
    os.makedirs(log_dir, exist_ok=True)

    # 1) Generate the training data
    training_data = generate_training_set(scale=scale, num_instances=100)
    
    # Create env
    env = ContainerStackEnv(training_data, max_height, max_steps)
    env = ActionMasker(env, mask_fn)
    # Save logs in `log_dir`
    env = Monitor(env, log_dir)

    
    model = MaskablePPO(
        policy= MaskableActorCriticPolicy,
        env=env,
        verbose = 1,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        tensorboard_log="tensorboard_log",
        policy_kwargs={
            "optimizer_class": th.optim.Adam,
            "optimizer_kwargs": dict(
                eps=1e-5,
                weight_decay=1e-5
            ),
        }
    )
    
    
    # Train with monitoring
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        
    )
    
    
    # Save model
    save_name = f"PPO_VANILLA_scale_act_mask_scale{scale}.zip"
    save_path = os.path.join(model_save_path, save_name)
    os.makedirs(model_save_path, exist_ok=True)
    model.save(save_path)
    
    env.close()
    return model, log_dir

def main(scale):
    model_save_dir = "models"
    model, log_dir = train_for_scale(
            scale=scale,
            max_height=3,
            max_steps=1000,
            total_timesteps=2000000,  
            model_save_path=model_save_dir

        )
   

 


if __name__ == "__main__":
    scale = 30
    main(scale)