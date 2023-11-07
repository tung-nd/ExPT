import numpy as np
import design_bench
import torch

from utils import TASK_ABBREVIATIONS, DISCRETE, NAME_TO_FULL_DATASET

def prepare_training_xs(task_name, normalize_x_scheme):
    if task_name != 'tf10':
        task = design_bench.make(TASK_ABBREVIATIONS[task_name])
        full_dataset = NAME_TO_FULL_DATASET[TASK_ABBREVIATIONS[task_name]]()
    else:
        task = design_bench.make(TASK_ABBREVIATIONS[task_name], dataset_kwargs={"max_samples": 10000})
        full_dataset = NAME_TO_FULL_DATASET[TASK_ABBREVIATIONS[task_name]](max_samples=50000)
    x_public = task.x
    x_all = full_dataset.x
    
    if task_name in DISCRETE:
        x_public = task.to_logits(x_public).reshape(x_public.shape[0], -1)
        x_all = task.to_logits(x_all).reshape(x_all.shape[0], -1)

    if normalize_x_scheme == 'scale':
        min_x = np.min(x_all, axis=0)
        max_x = np.max(x_all, axis=0)
        mean_x = min_x
        std_x = max_x - min_x
        std_x = np.where(std_x == 0, 1.0, std_x)
    elif normalize_x_scheme == 'standardize':
        mean_x = np.mean(x_public, axis=0)
        std_x = np.std(x_public, axis=0)
        std_x = np.where(std_x == 0, 1.0, std_x)
    elif normalize_x_scheme == 'none':
        mean_x = np.array([0.0], dtype=np.float32)
        std_x = np.array([1.0], dtype=np.float32)
    else:
        raise NotImplementedError

    x = x_public
    shuffled_ids = np.random.permutation(x.shape[0])
    x = x[shuffled_ids]
    x = torch.from_numpy(x)
    mean_x = torch.from_numpy(mean_x)
    std_x = torch.from_numpy(std_x)

    return x, mean_x, std_x

def prepare_eval_data(task_name, normalize_x_scheme, normalize_y_scheme):
    if task_name != 'tf10':
        task = design_bench.make(TASK_ABBREVIATIONS[task_name])
    else:
        task = design_bench.make(TASK_ABBREVIATIONS[task_name], dataset_kwargs={"max_samples": 10000})
    x = task.x
    y = task.y

    if task_name in DISCRETE:
        x = task.to_logits(x).reshape(x.shape[0], -1)

    if normalize_x_scheme == 'scale':
        if task_name != 'tf10':
            full_dataset = NAME_TO_FULL_DATASET[TASK_ABBREVIATIONS[task_name]]()
        else:
            full_dataset = NAME_TO_FULL_DATASET[TASK_ABBREVIATIONS[task_name]](max_samples=50000)
        x_full = full_dataset.x
        if task_name in DISCRETE:
            x_full = task.to_logits(x_full).reshape(x_full.shape[0], -1)
        min_x = np.min(x_full, axis=0)
        max_x = np.max(x_full, axis=0)
        mean_x = min_x
        std_x = max_x - min_x
        std_x = np.where(std_x == 0, 1.0, std_x)
    elif normalize_x_scheme == 'standardize':
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0)
        std_x = np.where(std_x == 0, 1.0, std_x)
    elif normalize_x_scheme == 'none':
        mean_x = np.array([0.0], dtype=np.float32)
        std_x = np.array([1.0], dtype=np.float32)
    else:
        raise NotImplementedError

    if normalize_y_scheme == 'scale':
        if task_name != 'tf10':
            full_dataset = NAME_TO_FULL_DATASET[TASK_ABBREVIATIONS[task_name]]()
        else:
            full_dataset = NAME_TO_FULL_DATASET[TASK_ABBREVIATIONS[task_name]](max_samples=50000)
        y_full = full_dataset.y
        min_y = np.min(y_full, axis=0)
        max_y = np.max(y_full, axis=0)
        mean_y = min_y
        std_y = max_y - min_y
    elif normalize_y_scheme == 'standardize':
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0)
    elif normalize_y_scheme == 'none':
        mean_y = np.array([0.0], dtype=np.float32)
        std_y = np.array([1.0], dtype=np.float32)
    else:
        raise NotImplementedError

    shuffled_ids = np.random.permutation(x.shape[0])
    x = x[shuffled_ids]
    y = y[shuffled_ids]

    x = torch.from_numpy(x)
    mean_x = torch.from_numpy(mean_x)
    std_x = torch.from_numpy(std_x)

    y = torch.from_numpy(y)
    mean_y = torch.from_numpy(mean_y)
    std_y = torch.from_numpy(std_y)

    return x, mean_x, std_x, y, mean_y, std_y