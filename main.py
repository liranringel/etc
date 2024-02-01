import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
from calib import run_calibration
from tsc_dataset import TscDataset
from quality_dataset import QualityDataset
from quality_model import QualityClassifier
from model import LSTMClassifier
from train import train
from utils import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_timesteps', type=int, default=200, help='Number of timesteps')
    parser.add_argument('--alpha', type=float, default=0.1, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size')
    parser.add_argument('--stop_threshold', type=float, default=1.0, help='Stop threshold')
    parser.add_argument('--is_correct_loss_factor', type=float, default=0.2, help='Is correct loss factor')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to model')
    parser.add_argument('--res_path', type=str, default='res.pt', help='Path to result')
    parser.add_argument('--dataset', type=str, default='Synthetic', help='Dataset name')
    parser.add_argument('--cal_type', type=str, default='normal', help='Calibration type')
    parser.add_argument('--accuracy_gap', type=float, default=0.1, help='Desired accuracy gap')
    parser.add_argument('--lambdas_step', type=float, default=0.01, help='Step size for lambda')
    parser.add_argument('--ltt_delta', type=float, default=0.01, help='Learn then Test delta')
    parser.add_argument('--train_only', action='store_true', help='Train only')

    args = parser.parse_args()
    assert args.hidden_size >= 3
    return args

def main():
    args = parse_args()
    set_global_seed(args.seed)

    if args.dataset == 'quality':
        ds = QualityDataset(args)
        args.n_timesteps = ds.n_timesteps
    else:
        ds = TscDataset(args.dataset)
        args.n_timesteps = ds.n_timesteps

    if args.dataset == 'quality':
        train_size = 0
        test_size = int(len(ds)/3)
        cal_size = len(ds) - train_size - test_size
        combined_test_cal_set = ds
    else:
        train_size = int(len(ds) * 0.7)
        val_size = int(len(ds) * 0.1)
        test_size = int(len(ds) * (0.2/3))
        cal_size = len(ds) - train_size - val_size - test_size
        combined_test_cal_size = test_size + cal_size
        torch.manual_seed(0)
        train_ds, val_ds, combined_test_cal_set = random_split(ds, [train_size, val_size, combined_test_cal_size])
    set_global_seed(args.seed)
    test_ds, cal_ds = random_split(combined_test_cal_set, [test_size, cal_size])
    
    if args.dataset == 'quality':
        quality_cache = None
        if os.path.exists('quality/quality_all_answers.pt'):
            all_answers = torch.load('quality/quality_all_answers.pt')
            quality_cache = {sample['id']: sample for sample in all_answers['X']}
        model = QualityClassifier(args, args.stop_threshold, cache=quality_cache)
    else:
        if os.path.exists(args.model_path):
            print('Loading model...')
            model = LSTMClassifier(args.num_layers, ds.input_size, args.hidden_size, ds.num_classes, args.stop_threshold)
            model.load_state_dict(torch.load(args.model_path))
        else:
            train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            model = LSTMClassifier(args.num_layers, ds.input_size, args.hidden_size, ds.num_classes, args.stop_threshold)
            print('Training model...')
            model = train(args, model, ds.n_timesteps, ds.num_classes, train_dl, val_dl)
            torch.save(model.state_dict(), args.model_path)
            print('Done training model')

    if args.train_only:
        return

    cal_X = [x for x, y in cal_ds]
    cal_y = [y for x, y in cal_ds]
    
    test_X = [x for x, y in test_ds]
    test_y = [y for x, y in test_ds]

    run_calibration(args, model, ds, cal_X, cal_y, test_X, test_y)


if __name__ == '__main__':
    main()
