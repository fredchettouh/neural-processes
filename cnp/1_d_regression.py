# custom imports
from .experiment import Experiment
from .datageneration import DataGenerator
frohelpers import Helper
# torch imports
from torch.utils import data
import torch
import os

# import utils
import json
from datetime import date

# we first get the variables for the configuration file
CHECKPOINT_DIR = os.environ['CHECKPOINT_DIR']
MODEL_PARAMS = os.environ['MODEL_PARAMS']

with open('configs/config.json') as f:
    file = f.read()
    conf = json.loads(file)

# we first generate the data to be used

datagenerator = DataGenerator(**conf[MODEL_PARAMS]['data_gen_params'])
x_values, func_x = datagenerator.generate_curves()
func_x = Helper.list_np_to_sensor(func_x)
x_values = x_values.repeat(func_x.shape[0], 1, 1)

# formatting the data as a dataloader

train_len = int(x_values.shape[0] * conf[MODEL_PARAMS]['train_share'])
traindata = data.TensorDataset(x_values[:train_len], func_x[:train_len])
trainloader = data.DataLoader(traindata, batch_size=10)
validata = data.TensorDataset(x_values[train_len:], func_x[train_len:])
valiloader = data.DataLoader(validata, batch_size=1)

# creating an instance of to orchestrate the training
trainer = Experiment(**conf[MODEL_PARAMS]['experiment_params'])
model_weights = trainer.run_training(trainloader, valiloader)

# exporting the weigths
current_date = date.today()
prefix = str(current_date.month) + str(current_date.day) + str(current_date.year)
suffix = '1_d_weights.pth'
file_name = CHECKPOINT_DIR + '/' + prefix + '_' + suffix
torch.save(model_weights, file_name)
