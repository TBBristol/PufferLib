

import wandb

RUN_NAME = None

old_init = wandb.init
def custom_init(*a, **k):
    if RUN_NAME:
        k.setdefault("name", RUN_NAME)
    return old_init(*a, **k)
wandb.init = custom_init


from pufferlib import pufferl
import glob, os

def latest_model_path(env_name):
    return max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)


if __name__ == '__main__':


    num_containers_list = [10,50,100,200]
    wandb_project = 'diayn_containers_varied'
    total_timesteps = 50000000
    for i in num_containers_list:

        max_ep_steps = i*2

    
        args = pufferl.load_config('puffer_stacking')
        args['env']['num_skills'] = 4
        args['env']['max_ep_steps'] = max_ep_steps
        args['env']['num_containers'] = i
        args['train']['total_timesteps'] = total_timesteps
        args['package'] = 'diayn_wrapper'
        args['vec']['backend'] = 'Serial' #REQURIED FOR NOW
        args['wandb'] = True
        #args['env']['device'] = 'cpu'
        args['wandb_project'] = wandb_project
        args['tag'] = 'diayn_num_containers_'+str(i)
        RUN_NAME = 'ST_Cont_'+str(i)
        args['train']['device'] = 'cuda'
        pufferl.train('puffer_stacking', args = args)





        args = pufferl.load_config('puffer_stacking')
        args['load_model_path'] = None
        args['env']['max_ep_steps'] = max_ep_steps
        args['env']['num_containers'] = i
        args['train']['total_timesteps'] = total_timesteps
        args['env']['diayn_model'] = latest_model_path('puffer_stacking')
        args['vec']['backend'] = 'Serial' #REQURIED FOR NOW
        args['env']['num_skills'] = 4
        args['package'] = 'diayn_meta_wrapper'
        args['wandb'] = True
        args['wandb_project'] = wandb_project
        RUN_NAME = 'MT_Cont_'+str(i)
        args['train']['device'] = 'cuda'


    
        pufferl.train('puffer_stacking', args = args)
