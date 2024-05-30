import wandb
from main import main_proc

sweep_config = {
    'method': 'grid'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric

parameters_dict = {
    'num_epochs':{
        'values': [50, 100]
    },
    'learning_rate': {
        'values': [0.0001, 0.001, 0.01]
    },
    'weight_decay': {
        'values': [0.1, 0.5]
    },
    'camera_list': {
        'values': [0, 1]
    }
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='CPR')

main_proc = main_proc()

wandb.agent(sweep_id, main_proc.main_proc, count=24)


