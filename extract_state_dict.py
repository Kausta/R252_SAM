import os
import sys
import torch
import pickle
import argparse
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract the state dict from a pl module')
    parser.add_argument('--file', default='', type=str)
    parser.add_argument('--save-to', default='', type=str)

    args = parser.parse_args()
    if args.file == '':
        raise Exception('Please provide file for model state.')
    if args.save_to == '':
        raise Exception('Please provide the file to save the state dict to.')
    
    ckpt = torch.load(args.file, map_location=torch.device('cpu'))
    with open(args.save_to, 'wb') as handle:
        pickle.dump({
                'state_dict': copy.deepcopy(ckpt['state_dict']), 
                'hparams': copy.deepcopy(ckpt['hyper_parameters'])
                }, file=handle) #saves the state dict and the config file
        print(f'Saved the state dict and config file to {args.save_to}')
