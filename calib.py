import torch
import numpy as np

from marginal_calibration import marginal_calibration
from conditional_calibration import conditional_calibration


def run_calibration(args, model, ds, cal_X, cal_y, test_X, test_y):
    if args.cal_type != 'marginal_accuracy_gap':
        half_len = len(cal_X) // 2
        cal2_X = cal_X[half_len:]
        cal2_y = cal_y[half_len:]
        cal_X = cal_X[:half_len]
        cal_y = cal_y[:half_len]

    model.stop_threshold = float('inf')
    late_res = model.forward(test_X)
    late_preds = late_res['scores'].detach().cpu().numpy().argmax(axis=-1)

    if args.cal_type == 'marginal_accuracy_gap':
        marginal_calibration(args, model, cal_X, cal_y)
    elif args.cal_type == 'conditional_accuracy_gap':
        conditional_calibration(args, model, cal_X, cal_y, cal2_X, cal2_y, skip_stage2=False)
    elif args.cal_type == 'conditional_without_stage2':
        conditional_calibration(args, model, cal_X, cal_y, cal2_X, cal2_y, skip_stage2=True)
    else:
        raise Exception(f'Unknown cal_type: {args.cal_type}')

    res = model.forward(test_X)
    preds = res['scores'].detach().cpu().numpy().argmax(axis=-1)
    halt_timesteps = res['halt_timesteps'].detach().cpu().numpy()

    is_correct = [test_y[i] == preds[i] for i in range(len(test_y))]
    late_is_correct = [test_y[i] == late_preds[i] for i in range(len(test_y))]
    accuracy = np.mean(is_correct)
    late_accuracy = np.mean(late_is_correct)
    print(f'{accuracy=} {late_accuracy=}')

    indices_by_t = [[] for _ in range(ds.n_timesteps)]
    for i in range(len(test_X)):
        indices_by_t[halt_timesteps[i]].append(i)

    t_accuracy_list = []
    t_num_correct_list = []
    t_late_accuracy_list = []
    t_late_num_correct_list = []
    t_num_samples_list = []
    t_gap_list = []
    for t in range(ds.n_timesteps):
        t_indices = indices_by_t[t]
        t_num_correct = np.sum([test_y[i] == preds[i] for i in t_indices])
        t_accuracy = np.mean([test_y[i] == preds[i] for i in t_indices])
        t_late_num_correct = np.sum([test_y[i] == late_preds[i] for i in t_indices])
        t_late_accuracy = np.mean([test_y[i] == late_preds[i] for i in t_indices])
        t_gap = np.sum([test_y[i] == late_preds[i] and test_y[i] != preds[i] for i in t_indices])
        t_gap_mean = t_gap / len(t_indices)
        print(f'{t=} {len(t_indices)=} {t_accuracy=:0.3f} {t_late_accuracy=:0.3f} {t_gap_mean=:0.3f}')
        t_accuracy_list.append(t_accuracy)
        t_late_accuracy_list.append(t_late_accuracy)
        t_num_correct_list.append(t_num_correct)
        t_late_num_correct_list.append(t_late_num_correct)
        t_num_samples_list.append(len(t_indices))
        t_gap_list.append(t_gap)
    
    to_save = {
        'stop_threshold': model.stop_threshold,
        'accuracy': accuracy,
        'late_accuracy': late_accuracy,
        't_accuracy_list': t_accuracy_list,
        't_late_accuracy_list': t_late_accuracy_list,
        't_num_correct_list': t_num_correct_list,
        't_late_num_correct_list': t_late_num_correct_list,
        't_num_samples_list': t_num_samples_list,
        'mean_halt_timesteps': np.mean(halt_timesteps),
        'halt_timesteps': halt_timesteps,
        't_gap_list': t_gap_list,
    }
    to_save2 = {
        'is_correct': is_correct,
        'late_is_correct': late_is_correct,
    }
    torch.save(to_save, args.res_path)
    torch.save(to_save2, args.res_path + '.2')
