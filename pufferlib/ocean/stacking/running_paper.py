from main import *
from stacking_paper_env import ContainerStackEnv
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

def get_possible_actions(obs):
        heights = obs[ :, 0]  # (B, n_stacks)
        # Convert to float for gradient propagation

        return (heights < 0.99) 

# 1) Generate the training data
scale = 30
training_data = generate_training_set(scale=scale, num_instances=100)

env = ContainerStackEnv(training_data, max_height = 3, max_steps = 100)

model = MaskablePPO.load("//Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/models/PPO_VANILLA_scale_act_mask_scale30.zip")

obs,info = env.reset(seed=1)
cum_reward = 0

for i in range(30):
    action_masks = get_action_masks(env)
    action = model.predict(obs, deterministic=True, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
   
    done = terminated or truncated
    if done:
         obs,info= env.reset()

    cum_reward += reward
env.render()




    

print(cum_reward)


obs,info = env.reset(seed=1)

cum_reward = 0
for i in range(30):

    mask = get_possible_actions(obs)
    true_indices = [i for i, val in enumerate(mask) if val]

    action = random.choice(true_indices)
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    if done:
        obs,info= env.reset()
    cum_reward += reward
env.render()
print(cum_reward)

    