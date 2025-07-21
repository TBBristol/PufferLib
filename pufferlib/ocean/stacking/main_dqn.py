import os
from stable_baselines3 import DQN
from stacking_paper_env import ContainerStackEnv
from stable_baselines3.common.monitor import Monitor
import random   

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
    # Save logs in `log_dir`
    env = Monitor(env, log_dir)

    
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
        
    )
    
    
    # Train with monitoring
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        
    )
    
    
    # Save model
    save_name = f"DQN_scale_{scale}.zip"
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
            total_timesteps=3000000,  
            model_save_path=model_save_dir

        )
   
    saved_model_path = f"{model_save_dir}/dqn_scale_{scale}.zip"
 


if __name__ == "__main__":
    scale = 10
    main(scale)