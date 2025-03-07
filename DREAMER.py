###############################################################################
# Imports
###############################################################################

import numpy as np
import torch
import tempfile
import os
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from models.EEGModels import EEGNet, EEGNet_SSVEP, EEGNet_ChanRed, EEGNet_WT, TCNet, TCNet_EMD
from preprocess.preprocess_DREAMER import DREAMERDataset
from tools import train_f, test_f, xDawnRG, margin_loss, MODEL_HYPEROPT_CHOICES, EMOTION_CHOICES

from torch.utils.data import DataLoader, SubsetRandomSampler

from ray.train import Checkpoint
from ray import tune, train
import ray
from ray.tune.schedulers import ASHAScheduler



def hyperopt_DREAMER(args):
      device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
      if device.type != "cuda":
            raise Exception("CUDA not available. Please select a GPU device.")
      else:
            print('Using device:', device)
      properties = torch.cuda.get_device_properties(device)
      n_cpu = os.cpu_count()
      n_gpu = torch.cuda.device_count()
      accelerator = properties.name.split()[1]
      n_parallel = 2

      
      ###############################################################################
      # Hyperparameters
      ###############################################################################

      selected_emotion = args.emotion
      selected_model = args.model
      is_ok = selected_model != 'TCNet' and device.type != 'mps' #TODO: understand why TCNet doesn't work with mixed precision (probably overflows)

      best_highcut = None
      best_lowcut = .5
      best_order = 3
      best_type = 'butter'
      best_start = 1
      best_sample = 128
      best_stds = {'EEGNet': True, 'TCNet': False}
      best_std = best_stds[selected_model]
      subjects = [i for i in range(2)] # None

      epochs = 1000
      test_split = .2

      best_lrs = {'EEGNet': 0.001, 'TCNet': 0.0001}
      best_lr = best_lrs[selected_model]
      best_batch_sizes = {'EEGNet': 128, 'TCNet': 64}
      best_batch_size = best_batch_sizes[selected_model]
      best_batch_size = 128
      best_F1 = 64
      best_D = 8
      best_F2 = 64
      best_kernLengths = {'EEGNet': {'arousal': 20, 'dominance': 12, 'valence': 12}, 'TCNet': {'arousal': 1, 'dominance': 12, 'valence': 8}} # perhaps go back to 64 for f_min = 2Hz
      best_kernLength = best_kernLengths[selected_model][selected_emotion]
      best_dropout = .1
      best_norm_rate = .25
      best_nr = 1.
      best_innerChanss = {'EEGNet': 18, 'TCNet': 192}
      best_innerChans = best_innerChanss[selected_model]
      best_tfr = {'emd':2}, # {'freqs': np.arange(2, 50), 'output': 'power'}
      best_num_heads = 4

      best_group_classes = True
      best_adapt_classWeights = False
      best_shifted = True

      chans = 14

      cur_dir = Path(__file__).resolve().parent
      figs_path = str(cur_dir) + '/figs/'
      sets_path = str(cur_dir) + '/sets/'
      models_path = str(cur_dir) + '/tmp/'
      save = False

      random_seed= 42
      np.random.seed(random_seed)
      num_s = 1


      ###############################################################################
      # Search for optimal hyperparameters
      ###############################################################################

      search_space = {
      "lr": best_lr, # tune.grid_search([.0001, .00001, .000001, .0000001]),
      "batch_size": best_batch_size, # tune.choice([32, 64, 128, 256, 512]),
      "sample": best_sample, # tune.grid_search([128, 256, 512, 1024, 2048]),
      "start": best_start, # tune.grid_search([0, 1, 2, 3, 4]),
      "order": best_order, # tune.grid_search([3, 5]),
      "lowcut": best_lowcut, # tune.grid_search([.5, .3]),
      "highcut": best_highcut, # tune.grid_search([None, 50, 55, 45]),
      "F1": best_F1, # tune.grid_search([16, 32, 64, 128, 256]),
      "D": best_D, # tune.grid_search([2, 4, 8, 16, 32]),
      "F2": best_F2, # tune.grid_search([4, 16, 64, 128, 256]),
      "kernLength": best_kernLength, # tune.grid_search([4, 8, 12, 16, 20, 24, 28, 32]),
      "dropout": best_dropout, # tune.grid_search([.1, .3]),
      "type": best_type, # tune.grid_search(["butter", "cheby1", "cheby2", "ellip", "bessel"])
      "group_classes": best_group_classes, # tune.grid_search([True, False]),
      "adapt_classWeights": best_adapt_classWeights, # tune.grid_search([True, False])
      "norm_rate": best_norm_rate, # tune.grid_search([.25, 1., None]),
      "nr": best_nr, #  tune.grid_search([.25, 1., None])
      "innerChans":  best_innerChans, # tune.grid_search([16, 18, 20]),
      "tfr": best_tfr, # tune.grid_search([{'emd':2}, {'eemd':2, 'sep_trends':True}, {'eemd':2, 'sep_trends':False}, {'ceemdan':2, 'beta_prog':True}, {'ceemdan':2, 'beta_prog':False}]),
      "std": best_std, # tune.grid_search([False, True]),
      "shifted": best_shifted, # tune.grid_search([True, False])
      "num_heads": best_num_heads # tune.grid_search([1, 2, 4, 8])
      }

      def train_DREAMER(config):

            info_str = 'DREAMER_' + selected_emotion + f'_subject({subjects})_filtered({config["lowcut"]}, {config["highcut"]}, {config["order"]})_samples({config["sample"]})_start({config["start"]})_'
            if config["group_classes"]:
                  class_weights = torch.tensor([1., 1.]).to(device)
                  nb_classes = 2
            else:
                  class_weights = torch.tensor([1., 1., 1., 1., 1.]).to(device)
                  nb_classes = 5

            ###############################################################################
            # Data loading
            ###############################################################################

            dataset = DREAMERDataset(sets_path+info_str, selected_emotion, subjects=subjects, sessions=None, samples=config["sample"], start=config["start"],
                                    lowcut=config["lowcut"], highcut=config["highcut"], order=config["order"], type=config["type"], save=save,
                                    group_classes=config["group_classes"], tfr=config["tfr"], use_ecg=False, std=config["std"], n_jobs=n_cpu)
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

            if selected_model == 'EEGNet':
                  model = EEGNet_WT(nb_classes=nb_classes, Chans=chans, InnerChans=config["innerChans"], Samples=config["sample"], dropoutRate=config['dropout'], 
                                    kernLength=config['kernLength'], F1=config['F1'], D=config['D'], F2=config['F2'], norm_rate=config["norm_rate"], nr=config["nr"],
                                    dropoutType='Dropout', nb_freqs=list(config["tfr"].values())[0]+1).to(device=device, memory_format=torch.channels_last)
            elif selected_model == 'TCNet':
                  model = TCNet_EMD(nb_classes=nb_classes, Chans=chans, nb_freqs=list(config["tfr"].values())[0]+1, shifted=config["shifted"],
                                    kern_emd=config["kernLength"], innerChans=config["innerChans"], num_heads=config["num_heads"]).to(device=device)

            loss_fn = torch.nn.CrossEntropyLoss(weight=dataset.class_weights).cuda() if config["adapt_classWeights"] else torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
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
            tune.with_resources(train_DREAMER, resources=tune.PlacementGroupFactory([{"CPU": n_cpu/n_parallel, "GPU": n_gpu/n_parallel, f"accelerator_type:{accelerator}": n_gpu/n_parallel}])),
            run_config=train.RunConfig(
                  checkpoint_config=train.CheckpointConfig(checkpoint_at_end=False, num_to_keep=4),
                  verbose=0
            ),
            tune_config=tune.TuneConfig(
                  metric="mean_accuracy",
                  mode="max",
                  num_samples=num_s,
                  scheduler=ASHAScheduler(max_t=epochs, grace_period=100),
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
      plt.title("Best config is: \n" + str(results.get_best_result().config), fontsize=10)
      plt.xlabel("Epochs")
      plt.ylabel("Mean Accuracy")
      plt.savefig(figs_path + 'tune_model_results.png')

      best_result = results.get_best_result("mean_accuracy", mode="max")
      with best_result.checkpoint.as_directory() as checkpoint_dir:
            state_dict = torch.load(os.path.join(checkpoint_dir, "model.pth"))
            torch.save(state_dict, models_path + "best_model.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimize hyperparameters with the DREAMER dataset.")
    parser.add_argument('model', choices=MODEL_HYPEROPT_CHOICES)
    parser.add_argument('emotion', choices=EMOTION_CHOICES)
    args = parser.parse_args()
    
    hyperopt_DREAMER(args)