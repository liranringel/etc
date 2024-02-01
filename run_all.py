import argparse
import subprocess
import concurrent.futures

TRAIN_MODELS_CONFIGS = {
    'Tiselac': '--dataset=Tiselac',
    'ElectricDevices': '--dataset=ElectricDevices',
    'PenDigits': '--dataset=PenDigits',
    'Crop': '--dataset=Crop',
    'WalkingSittingStanding': '--dataset=WalkingSittingStanding --hidden_size=256 --num_layers=2',
}

CALIBRATE_MODELS_CONFIGS = {
    'Tiselac': '--dataset=Tiselac',
    'ElectricDevices': '--dataset=ElectricDevices',
    'PenDigits': '--dataset=PenDigits',
    'Crop': '--dataset=Crop',
    'WalkingSittingStanding': '--dataset=WalkingSittingStanding --hidden_size=256 --num_layers=2',
    'quality': '--dataset=quality',
}

DATASET_TO_ACCURACY_GAP = {
    'Tiselac': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],
    'ElectricDevices': [0.1],
    'PenDigits': [0.1],
    'Crop': [0.1],
    'WalkingSittingStanding': [0.1],
    'quality': [0.1],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--seeds', type=int, default=100, help='Seeds to run')
    args = parser.parse_args()
    return args


def run_task(task):
    print(f'Running task: {task}')
    subprocess.run(task, shell=True)

def main():
    args = parse_args()
    max_workers = 5
    tasks_to_run = ['train_models', 'marginal_accuracy_gap', 'conditional_accuracy_gap', 'conditional_without_stage2']
    tasks = []

    if 'train_models' in tasks_to_run:
            for dataset in TRAIN_MODELS_CONFIGS:
                params = TRAIN_MODELS_CONFIGS[dataset]
                task = f'python main.py --model_path=checkpoints/dataset={dataset}.pt {params} --train_only'
                tasks.append(task)
    if 'marginal_accuracy_gap' in tasks_to_run:
        cal_type = 'marginal_accuracy_gap'
        for seed in range(args.seeds):
            for dataset in CALIBRATE_MODELS_CONFIGS:
                for accuracy_gap in DATASET_TO_ACCURACY_GAP[dataset]:
                    params = CALIBRATE_MODELS_CONFIGS[dataset]
                    task = f'python main.py --seed={seed} --model_path=checkpoints/dataset={dataset}.pt --res_path=results/dataset={dataset}_seed={seed}_cal_type={cal_type}_accuracy_gap={accuracy_gap}.pt --cal_type={cal_type} --accuracy_gap={accuracy_gap} {params}'
                    tasks.append(task)
    if 'conditional_accuracy_gap' in tasks_to_run:
        cal_type = 'conditional_accuracy_gap'
        for seed in range(args.seeds):
            for dataset in CALIBRATE_MODELS_CONFIGS:
                for accuracy_gap in DATASET_TO_ACCURACY_GAP[dataset]:
                    params = CALIBRATE_MODELS_CONFIGS[dataset]
                    task = f'python main.py --seed={seed} --model_path=checkpoints/dataset={dataset}.pt --res_path=results/dataset={dataset}_seed={seed}_cal_type={cal_type}_accuracy_gap={accuracy_gap}.pt --cal_type={cal_type} --accuracy_gap={accuracy_gap} {params}'
                    tasks.append(task)
    if 'conditional_without_stage2' in tasks_to_run:
        cal_type = 'conditional_without_stage2'
        for seed in range(args.seeds):
            for dataset in CALIBRATE_MODELS_CONFIGS:
                for accuracy_gap in DATASET_TO_ACCURACY_GAP[dataset]:
                    params = CALIBRATE_MODELS_CONFIGS[dataset]
                    task = f'python main.py --seed={seed} --model_path=checkpoints/dataset={dataset}.pt --res_path=results/dataset={dataset}_seed={seed}_cal_type={cal_type}_accuracy_gap={accuracy_gap}.pt --cal_type={cal_type} --accuracy_gap={accuracy_gap} {params}'
                    tasks.append(task)


    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the thread pool
        for task in tasks:
            executor.submit(run_task, task)

        # Wait for all tasks to complete
        executor.shutdown()


if __name__ == '__main__':
    main()
