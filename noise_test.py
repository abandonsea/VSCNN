#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:33 2021

@author: Pedro Vieira
@description: Implements the test function for the SAE-3DDRN network adding noise to the input image
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from net.vscnn_paviau import VSCNN
from test import test_model
from utils.config import VSCNNConfig
from utils.dataset import VSCNNDataset
from utils.noise import add_noise
from utils.tools import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################
# SET TEST CONFIG FILE #
########################
CONFIG_FILE = 'experiments/server_01/config.yaml'  # Empty string to load default 'config.yaml'
NOISES = [['salt_and_pepper', 0], ['salt_and_pepper', 0.005], ['salt_and_pepper', 0.01], ['salt_and_pepper', 0.05],
          ['additive_gaussian', 0.01], ['additive_gaussian', 0.05], ['additive_gaussian', 0.1],
          ['additive_gaussian', 0.2], ['additive_gaussian', 0.3],
          ['multiplicative_gaussian', 0], ['multiplicative_gaussian', 0.1], ['multiplicative_gaussian', 0.3],
          ['multiplicative_gaussian', 0.5], ['multiplicative_gaussian', 0.7]]


# Test SAE-3DDRN runs
def test():
    # Load config data from training
    config_file = 'config.yaml' if not CONFIG_FILE else CONFIG_FILE
    cfg = VSCNNConfig(config_file, test=True)

    # Set string modifier if testing best models
    test_best = 'best_' if cfg.test_best_models else ''
    if cfg.test_best_models:
        print('Testing best models from each run!')

    # Load raw dataset, apply PCA and normalize dataset.
    data = HSIData(cfg.dataset, cfg.data_folder, cfg.sample_bands)

    for run in range(cfg.num_runs):
        print(f'TESTING RUN {run + 1}/{cfg.num_runs}')

        _, test_gt, _ = HSIData.load_samples(cfg.split_folder, cfg.train_split, cfg.val_split, run)
        num_classes = len(np.unique(test_gt)) - 1  # Remove one for the "undefined" class

        for noise in NOISES:
            print(f'Using {noise[0]} noise with parameter: {noise[1]}')
            noisy_data = add_noise(data.raw_image, noise)
            pca_noisy_data = data.apply_transforms(noisy_data)

            # Load test ground truth and initialize test loader
            test_dataset = VSCNNDataset(pca_noisy_data, test_gt, cfg.sample_size, data_augmentation=False)
            test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

            # Load model
            model_file = f'{cfg.exec_folder}runs/vscnn_{test_best}model_run_{run}.pth'
            model = nn.DataParallel(VSCNN(cfg.sample_bands, num_classes))  # Values are going to be overwritten
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()

            # Set model to device
            model = model.to(device)

            # Test model from the current run
            report = test_model(model, test_loader)
            filename = cfg.results_folder + f'test_{noise[0]}.txt'
            save_results(filename, report, run)


# Main for running test independently
def main():
    test()


if __name__ == '__main__':
    main()
