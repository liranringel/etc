import torch
import numpy as np

from ltt import bin_p_value


def marginal_calibration(args, model, cal_X, cal_y):
    n_cal = len(cal_X)

    tmp_res = model.forward(cal_X)
    pi = tmp_res['all_is_correct_estimation'].detach().cpu()
    preds_per_t = tmp_res['all_scores'].detach().cpu().numpy().argmax(axis=-1)
    is_correct_per_t = np.array([int(cal_y[i]) == preds_per_t[i] for i in range(len(cal_y))])
    late_is_correct = is_correct_per_t[:, -1]

    lambda_hat = float('inf')
    lambda_ = 1.0
    
    while lambda_ >= 0:
        should_stop = pi >= lambda_
        should_stop[:, -1] = True  # Always stop at the last time step
        halt_timesteps = should_stop.float().argmax(dim=-1)
        is_correct = is_correct_per_t[torch.arange(n_cal), halt_timesteps]

        gap_sum = np.maximum(late_is_correct.astype(float) - is_correct.astype(float), 0).sum()
        p_value = bin_p_value(gap_sum, n_cal, args.accuracy_gap)

        if p_value > args.ltt_delta:
            break
        lambda_hat = lambda_
        lambda_ -= args.lambdas_step

    model.stop_threshold = lambda_hat
    print(f'Chose stop_threshold: {model.stop_threshold}')
    return model.stop_threshold
