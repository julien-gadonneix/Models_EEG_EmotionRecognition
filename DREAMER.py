###############################################################################
# Imports
###############################################################################

import numpy as np
import torch
import tempfile
import os
import matplotlib.pyplot as plt
from pathlib import Path

from models.EEGModels import EEGNet, EEGNet_SSVEP
from preprocess.preprocess_DREAMER import DREAMERDataset
from tools import train_f, test_f, xDawnRG, subject_dependent_classification_accuracy

from torch.utils.data import DataLoader, SubsetRandomSampler

from ray.train import Checkpoint
from ray import tune, train
import ray
from ray.tune.schedulers import ASHAScheduler


###############################################################################
# Hyperparameters
###############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', device)

best_highcut = None
best_lowcut = .5
best_order = 3
# lowcuts = [.5, .3]
# highcuts = [None, 60]
# orders = [3, 5]
best_start = 1
# starts = [0, 1, 2, 3, 4]
best_sample = 256
# samples = [128, 256, 512, 1024, 2048]
# subjects = [i for i in range(23)]
subject = None

epochs = 800
random_seed= 42
test_split = .33

best_lr = 0.001
best_batch_size = 128
best_F1 = 32
best_D = 8
best_F2 = 256
best_kernLength = 32 # maybe go back to 64 because now f_min = 4Hz
best_dropout = .3

selected_emotion = 'valence'
class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device)
names = ['1', '2', '3', '4', '5']
print('Selected emotion:', selected_emotion)

n_components = 2  # pick some components for xDawnRG
nb_classes = 5
chans = 14

cur_dir = Path(__file__).resolve().parent
figs_path = str(cur_dir) + '/figs/'
sets_path = str(cur_dir) + '/sets/'
models_path = str(cur_dir) + '/tmp/'

np.random.seed(random_seed)
num_s = 1
info_str = 'DREAMER_' + selected_emotion + f'_subject({subject})_filtered({best_lowcut}, {best_highcut}, {best_order})_samples({best_sample})_start({best_start})_'


###############################################################################
# Search for optimal hyperparameters
###############################################################################

search_space = {
    "lr": best_lr, # tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    "batch_size": best_batch_size, # tune.choice([32, 64, 128, 256, 512]),
    "F1": tune.grid_search([4, 8, 16, 32, 64]),
    "D": tune.grid_search([1, 2, 4, 8, 16]),
    "F2": tune.grid_search([4, 16, 64, 256, 1024]),
    "kernLength": tune.grid_search([16, 32, 64, 128]),
    "dropout": tune.grid_search([.1, .3, .5])
}

def train_DREAMER(config):

      ###############################################################################
      # Data loading
      ###############################################################################

      dataset = DREAMERDataset(sets_path+info_str, selected_emotion, subject=subject, samples=best_sample, start=best_start,
                              lowcut=best_lowcut, highcut=best_highcut, order=best_order)
      dataset_size = len(dataset)

      indices = list(range(dataset_size))
      np.random.shuffle(indices)
      split_test = int(np.floor(test_split * dataset_size))
      train_indices, test_indices = indices[split_test:], indices[:split_test]

      # Creating data samplers and loaders:
      train_sampler = SubsetRandomSampler(train_indices)
      test_sampler = SubsetRandomSampler(test_indices)
      train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler)
      test_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=test_sampler)

      print(len(train_indices), 'train samples')
      print(len(test_indices), 'test samples')


      ###############################################################################
      # Model configurations
      ###############################################################################

      model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=best_sample, 
                  dropoutRate=config['dropout'], kernLength=config['kernLength'], F1=config['F1'], D=config['D'], F2=config['F2'], dropoutType='Dropout').to(device)

      loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
      optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])


      ###############################################################################
      # Train and test
      ###############################################################################

      for epoch in range(epochs):
            train_f(model, train_loader, optimizer, loss_fn, device)
            acc = test_f(model, test_loader, device)

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                  checkpoint = None
                  if (epoch + 1) % 50 == 0:
                        torch.save(
                              model.state_dict(),
                              os.path.join(temp_checkpoint_dir, "model.pth")
                        )
                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                  # Send the current training result back to Tune
                  train.report({"mean_accuracy": acc}, checkpoint=checkpoint)

ray.init(num_cpus=16, num_gpus=1)
tuner = tune.Tuner(
    tune.with_resources(train_DREAMER, resources=tune.PlacementGroupFactory([{"CPU": 4, "GPU": .25, "accelerator_type:RTX": .25}])),
#     run_config=train.RunConfig(
#           stop={
#                 "mean_accuracy": 0.95,
#                 "training_iteration": num_s
#                 },
#                 checkpoint_config=train.CheckpointConfig(
#                       checkpoint_at_end=True, checkpoint_frequency=3
#                       )
#     ),
    tune_config=tune.TuneConfig(
          metric="mean_accuracy",
          mode="max",
          num_samples=num_s,
          scheduler=ASHAScheduler(max_t=epochs, grace_period=10)
    ),
    param_space=search_space
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)

dfs = {result.path: result.metrics_dataframe for result in results}
ax = None
for d in dfs.values():
      ax = d.mean_accuracy.plot(ax=ax, legend=False)
plt.title("Best config is: \n" + results.get_best_result().config, fontsize=10)
plt.xlabel("Epochs")
plt.ylabel("Mean Accuracy")
plt.savefig(figs_path + 'tune_model_results.png')

best_result = results.get_best_result("mean_accuracy", mode="max")
with best_result.checkpoint.as_directory() as checkpoint_dir:
    state_dict = torch.load(os.path.join(checkpoint_dir, "model.pth"))
    torch.save(state_dict, models_path + "best_model.pth")

###############################################################################
# Statistical benchmark analysis
###############################################################################

# xDawnRG(dataset, n_components, train_indices, test_indices, chans, samples, names, figs_path, info_str)


