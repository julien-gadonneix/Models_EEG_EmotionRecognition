###############################################################################
# Imports
###############################################################################

import numpy as np
import torch
import tempfile
import os
import matplotlib.pyplot as plt
from pathlib import Path

from models.EEGModels import EEGNet, EEGNet_SSVEP, EEGNet_ChanRed
from preprocess.preprocess_SEED import SEEDDataset
from tools import train_f, test_f, xDawnRG

from torch.utils.data import DataLoader, SubsetRandomSampler

from ray.train import Checkpoint
from ray import tune, train
import ray
from ray.tune.schedulers import ASHAScheduler


###############################################################################
# Hyperparameters
###############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
is_ok = device.type != 'mps'
if device.type != "cuda":
      raise Exception("CUDA not available. Please select a GPU device.")
else:
      print('Using device:', device)
properties = torch.cuda.get_device_properties(device)
n_cpu = os.cpu_count()
n_gpu = torch.cuda.device_count()
accelerator = properties.name.split()[1]
n_parallel = 2

best_start = 1
best_sample = 200
subjects = None
sessions = None

epochs = 300
random_seed= 42
test_split = .25

best_lr = 0.001
best_batch_size = 128
best_F1 = 64
best_D = 8
best_F2 = 64
best_kernLength = 15 # perhaps go back to 100 for f_min = 2Hz
best_dropout = .1
best_norm_rate = .25
best_nr = 1.

names = ['Negative', 'Neutral', 'Positive']

nb_classes = len(names)
chans = 62
best_innerChans = 24

cur_dir = Path(__file__).resolve().parent
figs_path = str(cur_dir) + '/figs/'
sets_path = str(cur_dir) + '/sets/'
models_path = str(cur_dir) + '/tmp/'
save = False

np.random.seed(random_seed)
num_s = 1


###############################################################################
# Search for optimal hyperparameters
###############################################################################

search_space = {
    "lr": best_lr, # tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    "batch_size": best_batch_size, # tune.choice([32, 64, 128, 256, 512]),
    "F1": best_F1, # tune.grid_search([16, 32, 64, 128]),
    "D": best_D, # tune.grid_search([2, 4, 8, 16]),
    "F2": best_F2, # tune.grid_search([16, 32, 64, 128]),
    "kernLength": best_kernLength, # tune.grid_search([10, 15, 20, 25, 30, 35, 40, 45, 50]),
    "dropout": best_dropout, # tune.grid_search([.1, .3])
    "innerChans": tune.grid_search([18, 20, 22, 24, 26, 28, 30, 32]),
}

def train_SEED(config):

      info_str = 'SEED_' + f'_subject({subjects})_samples({best_sample})_start({best_start})_'
      # info_str = 'DREAMER_' + selected_emotion + f'_subject({subjects})_filtered({config["lowcut"]}, {config["highcut"]}, {config["order"]})_samples({config["sample"]})_start({config["start"]})_'

      ###############################################################################
      # Data loading
      ###############################################################################

      dataset = SEEDDataset(sets_path+info_str, subjects=subjects, sessions=sessions, samples=best_sample, start=best_start, save=False)
      dataset_size = len(dataset)

      indices = list(range(dataset_size))
      np.random.shuffle(indices)
      split_test = int(np.floor(test_split * dataset_size))
      train_indices, test_indices = indices[split_test:], indices[:split_test]

      # Creating data samplers and loaders:
      train_sampler = SubsetRandomSampler(train_indices)
      test_sampler = SubsetRandomSampler(test_indices)
      train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=n_gpu*n_parallel, pin_memory=True)
      test_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=test_sampler, num_workers=n_gpu*n_parallel, pin_memory=True)

      print(len(train_indices), 'train samples')
      print(len(test_indices), 'test samples')


      ###############################################################################
      # Model configurations
      ###############################################################################

      model = EEGNet_ChanRed(nb_classes=nb_classes, Chans=chans, InnerChans=config["innerChans"], Samples=best_sample, dropoutRate=config['dropout'],
                             kernLength=config['kernLength'], F1=config['F1'], D=config['D'], F2=config['F2'], norm_rate=best_norm_rate, nr=best_nr,
                             dropoutType='Dropout').to(device=device, memory_format=torch.channels_last)

      loss_fn = torch.nn.CrossEntropyLoss().cuda()
      optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
      scaler = torch.cuda.amp.GradScaler()

      # torch.backends.cudnn.benchmark = True


      ###############################################################################
      # Train and test
      ###############################################################################

      for epoch in range(epochs):
            _ = train_f(model, train_loader, optimizer, loss_fn, scaler, device, is_ok)
            acc, _ = test_f(model, test_loader, loss_fn, device, is_ok)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                  checkpoint = None
                  if epoch + 1 == epochs:
                        torch.save(
                              model.state_dict(),
                              os.path.join(temp_checkpoint_dir, "model.pth")
                        )
                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                  train.report({"mean_accuracy": acc}, checkpoint=checkpoint)

ray.init(num_cpus=n_cpu, num_gpus=n_gpu)
tuner = tune.Tuner(
    tune.with_resources(train_SEED, resources=tune.PlacementGroupFactory([{"CPU": n_cpu/n_parallel, "GPU": n_gpu/n_parallel, f"accelerator_type:{accelerator}": n_gpu/n_parallel}])),
    run_config=train.RunConfig(
          checkpoint_config=train.CheckpointConfig(checkpoint_at_end=False, num_to_keep=4),
          verbose=0
    ),
    tune_config=tune.TuneConfig(
          metric="mean_accuracy",
          mode="max",
          num_samples=num_s,
          scheduler=ASHAScheduler(max_t=epochs, grace_period=10),
          trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
          trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}"
    ),
    param_space=search_space
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)

dfs = {result.path: result.metrics_dataframe for result in results}
ax = None
for d in dfs.values():
      ax = d.mean_accuracy.plot(ax=ax, legend=False)
plt.title("Best config is: \n" + str(results.get_best_result().config), fontsize=8)
plt.xlabel("Epochs")
plt.ylabel("Mean Accuracy")
plt.savefig(figs_path + 'tune_model_results.png')

best_result = results.get_best_result("mean_accuracy", mode="max")
with best_result.checkpoint.as_directory() as checkpoint_dir:
    state_dict = torch.load(os.path.join(checkpoint_dir, "model.pth"))
    torch.save(state_dict, models_path + "best_model.pth")


