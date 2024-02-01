import torch
import numpy as np

from ltt import bin_p_value


def candidate_screening(args, model, screening_X, screening_y):
    n_screening = len(screening_X)
    tmp_res = model.forward(screening_X)
    pi = tmp_res['all_is_correct_estimation'].detach().cpu()
    preds_per_t = tmp_res['all_scores'].detach().cpu().numpy().argmax(axis=-1)
    is_correct_per_t = np.array([int(screening_y[i]) == preds_per_t[i] for i in range(len(screening_y))])
    late_is_correct = is_correct_per_t[:, -1]

    lambda_array = np.arange(0, 1.00001, args.lambdas_step)
    t_max = args.n_timesteps
    eta_hat = torch.ones(t_max) * float('inf')

    for t in range(t_max):
        eta = eta_hat.clone()
        for xi in lambda_array:
            eta[t] = xi
            should_stop = pi >= eta
            should_stop[:, -1] = True  # Always stop at the last time step
            halt_timesteps = should_stop.float().argmax(dim=-1)  # tau
            is_correct = is_correct_per_t[torch.arange(n_screening), halt_timesteps]
            I = torch.nonzero(halt_timesteps <= t).squeeze(dim=-1)

            if len(I) == 0:
                break
            
            gap = np.maximum(late_is_correct[I].astype(float) - is_correct[I].astype(float), 0).mean()
            if gap <= args.accuracy_gap:
                eta_hat[t] = xi
                break

    return eta_hat


def testing(args, model, cal2_X, cal2_y, eta_hat):
    t_max = args.n_timesteps
    n_testing = len(cal2_X)

    tmp_res = model.forward(cal2_X)
    pi = tmp_res['all_is_correct_estimation'].detach().cpu()
    preds_per_t = tmp_res['all_scores'].detach().cpu().numpy().argmax(axis=-1)
    is_correct_per_t = np.array([int(cal2_y[i]) == preds_per_t[i] for i in range(len(cal2_y))])
    late_is_correct = is_correct_per_t[:, -1]

    lambda_hat = torch.ones(t_max) * float('inf')
    stop_testing = False
    
    for t in range(t_max)[::-1]:
        lambda_ = lambda_hat.clone()
        lambda_[t] = eta_hat[t]
        for t_tag in range(t, t_max):
            should_stop = pi >= lambda_
            should_stop[:, -1] = True  # Always stop at the last time step
            halt_timesteps = should_stop.float().argmax(dim=-1)
            is_correct = is_correct_per_t[torch.arange(n_testing), halt_timesteps]
            I = torch.nonzero(halt_timesteps <= t_tag).squeeze(dim=-1)

            if len(I) == 0:
                stop_testing = True  # No evidence to reject the null
                break

            gap_sum = np.maximum(late_is_correct[I].astype(float) - is_correct[I].astype(float), 0).sum()
            p_value = bin_p_value(gap_sum, len(I), args.accuracy_gap)

            if p_value > args.ltt_delta:
                stop_testing = True
                break
        if stop_testing:
            break
        lambda_hat = lambda_.clone()

    model.stop_threshold = lambda_hat
    print(f'Chose stop_threshold: {model.stop_threshold}')
    return model.stop_threshold


def conditional_calibration(args, model, screening_X, screening_y, testing_X, testing_y, skip_stage2=False):
    eta_hat = candidate_screening(args, model, screening_X, screening_y)
    if skip_stage2:
        model.stop_threshold = eta_hat
        return eta_hat
    return testing(args, model, testing_X, testing_y, eta_hat)
