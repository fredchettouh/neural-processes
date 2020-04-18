# custom imports
from experiment import Experiment
from datageneration import DataGenerator
from helpers import Helper
# torch imports
from torch.utils import data
import torch

# import utils
import json
from datetime import date

BASE_DIR = 'checkpoints'

with open('config.json') as f:
    file = f.read()
    conf = json.loads(file)



datagenerator = DataGenerator(**conf['default']['data_gen_params'])
x_values, func_x = datagenerator.generate_curves()
func_x = Helper.list_np_to_sensor(func_x)
x_values = x_values.repeat(func_x.shape[0], 1, 1)

train_len = int(x_values.shape[0] * conf['default']['train_share'])
traindata = data.TensorDataset(x_values[:train_len], func_x[:train_len])
trainloader = data.DataLoader(traindata, batch_size=10)
validata = data.TensorDataset(x_values[train_len:], func_x[train_len:])
valiloader = data.DataLoader(validata, batch_size=1)

trainer = Experiment(**conf['default']['experiment_params'])

model_weights = trainer.run_training(trainloader,valiloader)

current_date = date.today()
prefix = str(current_date.month) + str(current_date.day) + str(current_date.year)
suffix = '1_d_weights.pth'
file_name = BASE_DIR + '/' + prefix + '_' + suffix

torch.save(model_weights, file_name)
