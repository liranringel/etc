import argparse

import torch

from quality_dataset import QualityDataset
from quality_model import QualityClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--n_timesteps', type=int, default=200, help='Number of timesteps')
    parser.add_argument('--part', type=int, required=True, help='Part of the dataset')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    part = args.part

    print('Loading dataset')
    ds = QualityDataset(args)
    print('Done loading dataset')
    args.n_timesteps = ds.n_timesteps
    model = QualityClassifier(args, 1.0)

    part_size = 1000
    X = [x for x, y in ds][part*part_size:(part+1)*part_size]
    y = [y for x, y in ds][part*part_size:(part+1)*part_size]
    res = model.forward(X)
    for i in range(len(X)):
        X[i]['all_scores'] = res['all_scores'][i]
        X[i]['all_is_correct_estimation'] = res['all_is_correct_estimation'][i]
    torch.save({
        'X': X,
        'y': y,
    }, f'quality/quality_answers_part{part}.pt')


if __name__ == '__main__':
    main()
