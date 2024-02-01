from functools import cache

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import seaborn as sns
import pandas as pd


plt.rcParams.update({'font.size': 21})


SEEDS = list(range(100))

DESIRED_GAP = 0.1
DPI = 300

DATASET_TO_MAX_LEN = {
    'Tiselac': 23,
    'ElectricDevices': 96,
    'PenDigits': 8,
    'Crop': 46,
    'WalkingSittingStanding': 206,
    'quality': 10,
}


@cache
def load_res(path: str):
    return torch.load(path)

@cache
def get_accuracy_gap_df(dataset: str, cal_type: str, accuracy_gap: float):
    accuracy_list = []
    late_accuracy_list = []
    t_accuracy_list_list = []
    t_late_accuracy_list_list = []
    t_num_correct_list_list = []
    t_late_num_correct_list_list = []
    t_num_samples_list_list = []
    t_gap_list_list = []
    halt_timesteps_list_list = []

    for seed in SEEDS:
        res_path = f'results/dataset={dataset}_seed={seed}_cal_type={cal_type}_accuracy_gap={accuracy_gap}.pt'
        res = load_res(res_path)
        accuracy_list.append(res['accuracy'])
        late_accuracy_list.append(res['late_accuracy'])
        t_accuracy_list_list.append(res['t_accuracy_list'])
        t_late_accuracy_list_list.append(res['t_late_accuracy_list'])
        t_num_correct_list_list.append(res['t_num_correct_list'])
        t_late_num_correct_list_list.append(res['t_late_num_correct_list'])
        t_num_samples_list_list.append(res['t_num_samples_list'])
        t_gap_list_list.append(res['t_gap_list'])
        halt_timesteps_list_list.append(res['halt_timesteps'])

    t_gap_list_list = np.array(t_gap_list_list)
    halt_timesteps_list_list = np.array(halt_timesteps_list_list)

    t_accuracy_list_list = np.array(t_accuracy_list_list)
    t_late_accuracy_list_list = np.array(t_late_accuracy_list_list)
    t_num_samples_list_list = np.array(t_num_samples_list_list)
    t_num_correct_list_list = np.array(t_num_correct_list_list)
    t_late_num_correct_list_list = np.array(t_late_num_correct_list_list)

    num_samples_until_t = t_num_samples_list_list.cumsum(axis=1)

    t_gap_until_t = t_gap_list_list.cumsum(axis=1)
    num_accuracy_gap_until_t = t_gap_until_t / num_samples_until_t

    num_timesteps = DATASET_TO_MAX_LEN[dataset]
    timesteps = np.tile(np.arange(1, num_timesteps+1), len(SEEDS))
    seeds = np.repeat(np.arange(1, len(SEEDS)+1), num_timesteps)
    values = num_accuracy_gap_until_t.flatten()

    data = {
        'TimeStep': timesteps,
        'Seed': seeds,
        'Value': values,
        'NumHalted': t_num_samples_list_list.flatten(),
        'NumHaltedUntilT': num_samples_until_t.flatten(),
    }
    df = pd.DataFrame(data)
    return df

def plot_marginal_vs_conditional(dataset: str):
    marginal_df = get_accuracy_gap_df(dataset, 'marginal_accuracy_gap', accuracy_gap=DESIRED_GAP)
    conditional_df = get_accuracy_gap_df(dataset, 'conditional_accuracy_gap', accuracy_gap=DESIRED_GAP)
    marginal_df['Experiment'] = 'Marginal'
    conditional_df['Experiment'] = 'Conditional'
    combined_df = pd.concat([marginal_df, conditional_df], ignore_index=True)

    max_timestep = DATASET_TO_MAX_LEN[dataset]
    first_tick = 1
    tick_spacing = max_timestep // 5
    xticks = [first_tick] + list(range(first_tick + tick_spacing, max_timestep+1, tick_spacing))
    xticks[-1] = max_timestep

    conditional_accuracy_gap_fig, axs = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1, 1]}, constrained_layout=True)
    pos1 = axs[0].get_position()  # Get the original position
    axs[0].set_position([pos1.x0, pos1.y0, pos1.width, pos1.height])
    pos2 = axs[1].get_position()  # Get the original position
    axs[1].set_position([pos2.x0+0.1, pos2.y0, pos2.width, pos2.height])
    ax = axs[0]
    hue_order = ['Marginal', 'Conditional']
    df_dashed = combined_df[combined_df['Experiment'] == 'Marginal']
    df_solid = combined_df[combined_df['Experiment'] != 'Marginal']
    sns.lineplot(data=df_dashed, x='TimeStep', y='Value', hue='Experiment', units='Seed', estimator=None, lw=1, alpha=0.2, ax=ax, linestyle='--', hue_order=hue_order)
    sns.lineplot(data=df_solid, x='TimeStep', y='Value', hue='Experiment', units='Seed', estimator=None, lw=1, alpha=0.2, ax=ax, linestyle='-', hue_order=hue_order)
    ax.set_xlim([1-max_timestep*(1/20), max_timestep*(21/20)])
    ax.set_xticks(xticks)
    desired_accuracy_gap = DESIRED_GAP
    if desired_accuracy_gap is not None:
        ax.axhline(y=DESIRED_GAP, color='r', linestyle=':')
        # make a legend for the red line
        ax.plot([], [], color='r', linestyle=':', label=f'$\\alpha$ level')
        ax.legend(borderaxespad=0.2, borderpad=0.25)
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Conditional accuracy gap')

    if dataset == 'quality':
        ax.set_ylim([0, 0.2])
    else:
        ax.set_ylim([0, 0.35])

    last_timestep = DATASET_TO_MAX_LEN[dataset]
    # Adding a vertical line on the last timestep
    ax.axvline(x=last_timestep, color='black', linestyle='--')

    text_y_position = ax.get_ylim()[1] * 0.87
    if dataset == 'quality':
        text_x_position = last_timestep*0.71
    elif dataset == 'Tiselac':
        text_x_position = last_timestep*0.70
        text_y_position = ax.get_ylim()[1] * 0.44
    elif dataset in {'Crop', 'PhonemeSpectra', 'ElectricDevices'}:
        text_x_position = last_timestep*0.69
    elif dataset == 'PenDigits':
        text_x_position = last_timestep*0.72
    elif dataset == 'WalkingSittingStanding':
        text_x_position = last_timestep*0.69
    ax.text(text_x_position, text_y_position, 'Marginal risk', color='black', ha='center', va='bottom')
    if dataset in {'quality', 'Tiselac'}:
        handles, labels = ax.get_legend_handles_labels()
        if dataset == 'quality':
            handles = handles[4:]
            labels = labels[4:]
        else:
            handles = handles[2:]
            labels = labels[2:]
            handles[0].set_linestyle('--')
        # Put the legend in the bottom left
        if dataset == 'quality':
            ax.legend(handles, labels, loc='lower left', borderaxespad=0.2, borderpad=0.25)
        else:
            loc = 'upper left'
            ax.legend(handles, labels, loc='upper left', borderaxespad=0.2, borderpad=0.25)
    else:
        ax.get_legend().remove()
    
    pos = ax.yaxis.label.get_position()
    ax.yaxis.label.set_position((pos[0], pos[1]-0.065))
    


    ax = axs[1]
    sns.lineplot(data=df_dashed, x='TimeStep', y='NumHaltedUntilT', hue='Experiment', errorbar=('ci', 95), ax=ax, hue_order=hue_order, linewidth=3, linestyle='--')
    sns.lineplot(data=df_solid, x='TimeStep', y='NumHaltedUntilT', hue='Experiment', errorbar=('ci', 95), ax=ax, hue_order=hue_order, linewidth=3, linestyle='-')
    ax.set_xlim([1-max_timestep*(1/20), max_timestep*(21/20)])
    ax.set_xticks(xticks)
    
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('#Samples with\nhalt time $\\hat{\\tau}(X) \leq t$')

    # Get current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Remove the first handle and label (which is typically the hue name)
    handles = handles[2:]
    labels = labels[2:]
    handles[0].set_linestyle('--')

    # Recreate the legend without the hue name
    ax.legend(handles, labels, borderaxespad=0.2, borderpad=0.25)
    # remove the legend
    if dataset not in {'quality'}:
        ax.get_legend().remove()

    return conditional_accuracy_gap_fig


def plot_stage1_vs_stage2(dataset: str):
    stage1_df = get_accuracy_gap_df(dataset, 'conditional_without_stage2', accuracy_gap=DESIRED_GAP)
    stage2_df = get_accuracy_gap_df(dataset, 'conditional_accuracy_gap', accuracy_gap=DESIRED_GAP)
    stage1_df['Experiment'] = 'Only Stage 1'
    stage2_df['Experiment'] = 'Conditional Method: Stage 1+2'
    combined_df = pd.concat([stage1_df, stage2_df], ignore_index=True)

    # plot accuracy gap
    conditional_accuracy_gap_fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    hue_order = ['Only Stage 1', 'Conditional Method: Stage 1+2']
    default_palette = sns.color_palette()
    orange_color = default_palette[1]
    palette = ['#404040', orange_color]
    df_dashed = combined_df[combined_df['Experiment'] == 'Only Stage 1']
    df_solid = combined_df[combined_df['Experiment'] != 'Only Stage 1']
    sns.lineplot(data=df_dashed, x='TimeStep', y='Value', hue='Experiment', units='Seed', estimator=None, lw=1, alpha=0.2, ax=ax, linestyle='--', hue_order=hue_order, palette=[palette[0]])
    sns.lineplot(data=df_solid, x='TimeStep', y='Value', hue='Experiment', units='Seed', estimator=None, lw=1, alpha=0.2, ax=ax, linestyle='-', hue_order=hue_order, palette=[palette[1]])
    desired_accuracy_gap = DESIRED_GAP
    if desired_accuracy_gap is not None:
        ax.axhline(y=DESIRED_GAP, color='r', linestyle=':')
        # make a legend for the red line
        ax.plot([], [], color='r', linestyle=':', label=f'$\\alpha$ level')
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Conditional accuracy gap')

    if dataset == 'quality':
        ax.set_ylim([0, 0.2])
    else:
        ax.set_ylim([0, 0.35])


    # Get current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Remove the first handle and label (which is typically the hue name)
    handles = [handles[0], handles[3], handles[-1]]
    labels = [labels[0], labels[3], '$\\alpha$ level']
    handles[0].set_linestyle('--')

    # Recreate the legend without the hue name
    plt.legend(handles, labels, borderaxespad=0.1, borderpad=0.25)

    return conditional_accuracy_gap_fig


def calc_cond_acc_gap_until_quantile_of_halt_time(dataset: str, cal_type: str, accuracy_gap: float, quantile: float):
    is_correct_list_list = []
    late_is_correct_list_list = []
    halt_timesteps_list_list = []
    for seed in SEEDS:
        res_path = f'results/dataset={dataset}_seed={seed}_cal_type={cal_type}_accuracy_gap={accuracy_gap}.pt'
        res_path2 = f'results/dataset={dataset}_seed={seed}_cal_type={cal_type}_accuracy_gap={accuracy_gap}.pt.2'
        res = load_res(res_path)
        res2 = load_res(res_path2)
        is_correct_list_list.append(res2['is_correct'])
        late_is_correct_list_list.append(res2['late_is_correct'])
        halt_timesteps_list_list.append(res['halt_timesteps'])

    is_correct_list_list = np.array(is_correct_list_list)
    late_is_correct_list_list = np.array(late_is_correct_list_list)
    halt_timesteps_list_list = np.array(halt_timesteps_list_list)

    # sort indices by halt time
    indices_by_halt_time = np.argsort(halt_timesteps_list_list, axis=1)
    indices_before_quantile = indices_by_halt_time[:, :int(len(indices_by_halt_time[0]) * quantile)]

    is_correct_before_quantile = is_correct_list_list[np.arange(len(indices_before_quantile))[:, None], indices_before_quantile]
    late_is_correct_before_quantile = late_is_correct_list_list[np.arange(len(indices_before_quantile))[:, None], indices_before_quantile]

    before_agg = np.maximum(late_is_correct_before_quantile.astype(float) - is_correct_before_quantile.astype(float), 0).mean(axis=1)
    accuracy_gap_before_quantile = before_agg.mean()
    accuracy_gap_before_quantile_std = before_agg.std() / np.sqrt(len(before_agg))
    return accuracy_gap_before_quantile, accuracy_gap_before_quantile_std

def calc_time_used_and_accuracy(dataset: str, cal_type: str, accuracy_gap: float):
    mean_halt_time_list = []
    accuracy_list = []
    late_accuracy_list = []

    for seed in SEEDS:
        res_path = f'results/dataset={dataset}_seed={seed}_cal_type={cal_type}_accuracy_gap={accuracy_gap}.pt'
        res = load_res(res_path)
        mean_halt_time_list.append(res['mean_halt_timesteps'])
        accuracy_list.append(res['accuracy'])
        late_accuracy_list.append(res['late_accuracy'])
    mean_halt_time_array = (np.array(mean_halt_time_list) + 1)/DATASET_TO_MAX_LEN[dataset]
    accuracy_array = np.array(accuracy_list)
    late_accuracy_array = np.array(late_accuracy_list)

    time_used_mean = np.mean(mean_halt_time_array)
    time_used_std_err = np.std(mean_halt_time_array) / np.sqrt(len(mean_halt_time_array))
    accuracy_mean = np.mean(accuracy_array)
    accuracy_std_err = np.std(accuracy_array) / np.sqrt(len(accuracy_array))
    late_accuracy_mean = np.mean(late_accuracy_array)
    late_accuracy_std_err = np.std(late_accuracy_array) / np.sqrt(len(late_accuracy_array))

    return time_used_mean, time_used_std_err, accuracy_mean, accuracy_std_err, late_accuracy_mean, late_accuracy_std_err


def get_t_avg_of_accuracy_gap(dataset: str, cal_type: str, accuracy_gap: float):
    mean_halt_time_list = []

    for seed in SEEDS:
        res_path = f'results/dataset={dataset}_seed={seed}_cal_type={cal_type}_accuracy_gap={accuracy_gap}.pt'
        res = load_res(res_path)
        mean_halt_time_list.append(res['mean_halt_timesteps'])
    mean_halt_time_array = (np.array(mean_halt_time_list) + 1)/DATASET_TO_MAX_LEN[dataset]

    return mean_halt_time_array

def plot_t_avg_of_accuracy_gap_tiselac():
    accuracy_gaps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    t_avg_array_list = []
    for accuracy_gap in accuracy_gaps:
        t_avg_array = get_t_avg_of_accuracy_gap('Tiselac', 'conditional_accuracy_gap', accuracy_gap)
        t_avg_array_list.append(t_avg_array)
    
    t_avg_array_array = np.array(t_avg_array_list)

    # t_avg_array_array is of size (len(accuracy_gaps), len(SEEDS))
    data = {
        'Accuracy Gap': np.repeat(accuracy_gaps, len(SEEDS)),
        'Seed': np.tile(SEEDS, len(accuracy_gaps)),
        't_avg': t_avg_array_array.flatten()
    }

    df = pd.DataFrame(data)

    plt.rcParams.update({'font.size': 12})
    
    t_avg_of_accuracy_gap_fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    sns.lineplot(x='Accuracy Gap', y='t_avg', data=df, marker='o', ax=ax, errorbar='se', err_style='bars')
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$T_{\\text{avg}}$')
    # change the font size of the x and y titles
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.set_ylim([0, 1])

    tick_spacing = 0.02
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))


    return t_avg_of_accuracy_gap_fig


def get_latex_table():
    dataset_list = []
    cal_type_list = []
    quantile_20_list = []
    quantile_50_list = []
    quantile_100_list = []
    quantile_20_std_err_list = []
    quantile_50_std_err_list = []
    quantile_100_std_err_list = []
    time_used_list = []
    time_used_std_err_list = []
    accuracy_list = []
    accuracy_std_err_list = []
    late_accuracy_list = []
    late_accuracy_std_err_list = []

    for dataset in DATASET_TO_MAX_LEN:
        for cal_type in ['marginal_accuracy_gap', 'conditional_accuracy_gap']:
            accuracy_gap_before_quantile20, accuracy_gap_before_quantile20_std = calc_cond_acc_gap_until_quantile_of_halt_time(dataset, cal_type, DESIRED_GAP, 0.2)
            accuracy_gap_before_quantile50, accuracy_gap_before_quantile50_std = calc_cond_acc_gap_until_quantile_of_halt_time(dataset, cal_type, DESIRED_GAP, 0.5)
            accuracy_gap_before_quantile100, accuracy_gap_before_quantile100_std = calc_cond_acc_gap_until_quantile_of_halt_time(dataset, cal_type, DESIRED_GAP, 1.0)
            dataset_list.append(dataset)
            cal_type_list.append(cal_type)
            quantile_20_list.append(accuracy_gap_before_quantile20)
            quantile_50_list.append(accuracy_gap_before_quantile50)
            quantile_100_list.append(accuracy_gap_before_quantile100)
            quantile_20_std_err_list.append(accuracy_gap_before_quantile20_std)
            quantile_50_std_err_list.append(accuracy_gap_before_quantile50_std)
            quantile_100_std_err_list.append(accuracy_gap_before_quantile100_std)
            time_used, time_used_std_err, accuracy_mean, accuracy_std_err, late_accuracy_mean, late_accuracy_std_err = calc_time_used_and_accuracy(dataset, cal_type, DESIRED_GAP)
            time_used_list.append(time_used)
            time_used_std_err_list.append(time_used_std_err)
            accuracy_list.append(accuracy_mean)
            accuracy_std_err_list.append(accuracy_std_err)
            late_accuracy_list.append(late_accuracy_mean)
            late_accuracy_std_err_list.append(late_accuracy_std_err)

    df = pd.DataFrame({
        'Dataset': dataset_list,
        'Calibration Type': cal_type_list,
        'Accuracy Gap 20% first': quantile_20_list,
        'Accuracy Gap 20% first Std Err': quantile_20_std_err_list,
        'Accuracy Gap 50% first': quantile_50_list,
        'Accuracy Gap 50% first Std Err': quantile_50_std_err_list,
        'Accuracy Gap 100% first': quantile_100_list,
        'Accuracy Gap 100% first Std Err': quantile_100_std_err_list,
        'Time Used': time_used_list,
        'Time Used Err': time_used_std_err_list,
        'Accuracy': accuracy_list,
        'Accuracy Err': accuracy_std_err_list,
        'Late Accuracy': late_accuracy_list,
        'Late Accuracy Err': late_accuracy_std_err_list,
    })
    # print latex table, each cell has the accuracy gap
    latex = df.to_latex(index=False, float_format='%.3f')
    return latex

def main():
    print('Writing latex table')
    latex = get_latex_table()
    with open('figures/accuracy_gap.tex', 'w') as f:
        f.write(latex)

    print('Plotting marginal_vs_conditional')
    for dataset in DATASET_TO_MAX_LEN:
        conditional_accuracy_gap_fig = plot_marginal_vs_conditional(dataset)
        conditional_accuracy_gap_fig.savefig(f'figures/{dataset}_marginal_vs_conditional_accuracy_gap.png', bbox_inches='tight', dpi=DPI)

    print('Plotting stage1_vs_stage2')
    stage1_vs_stage2_fig = plot_stage1_vs_stage2('quality')
    stage1_vs_stage2_fig.savefig(f'figures/quality_stage1_vs_stage2.png', bbox_inches='tight', dpi=DPI)
    
    print('Plotting t_avg_of_accuracy_gap_tiselac')
    plot_t_avg_of_accuracy_gap_tiselac().savefig(f'figures/t_avg_of_accuracy_gap_tiselac.png', bbox_inches='tight', dpi=DPI)

if __name__ == '__main__':
    main()
