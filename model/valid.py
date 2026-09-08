"""
    @Author: Su changwei
    @Email: scw727@outlook.com
    @Date: 2026-07-26
    @Description: Test script for TP 2D field reconstruction.
    @File: valid.py
"""

import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MISOReconstructionDataset
from model import DilatedSEReconstructionNet


try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required. Please install it using: pip install pyyaml"
    ) from exc


# 配置文件路径
config_path_list = [
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading0.yaml",]
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading2.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading4.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading6.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading8.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading10.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading12.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading14.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading16.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading18.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading20.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading22.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading24.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading26.yaml",
    # "C:/Users/sysu/Desktop/tp_reconstruction/config/leading28.yaml",]


def load_config(file_path):
    """
    读取 YAML 配置文件。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Config file does not exist: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def set_seed(seed):
    """
    设置随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir):
    """
    设置日志，同时输出到控制台和日志文件。
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"test_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return log_path


def build_dataloader(dataset, batch_size, num_workers, device):
    """
    构建验证数据 DataLoader。
    """
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(**loader_kwargs)


def load_checkpoint(
    checkpoint_path,
    model,
    device,
    channel_names,
    lead_days,
    current_input_mode,
):
    """
    加载训练保存的 .tar checkpoint，并检查通道顺序和 lead_days。
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    # 兼容不同 PyTorch 版本
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a dictionary, but got: {type(checkpoint)}"
        )

    if "state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain the key 'state_dict'."
        )

    saved_channel_names = checkpoint.get("channel_names")

    if saved_channel_names is not None:
        saved_channel_names = list(saved_channel_names)

        if saved_channel_names != list(channel_names):
            raise ValueError(
                "Input channel order does not match the checkpoint.\n"
                f"Checkpoint channels: {saved_channel_names}\n"
                f"Current channels:    {list(channel_names)}"
            )

    saved_lead_days = checkpoint.get("lead_days")

    if saved_lead_days is not None:
        saved_lead_days = int(saved_lead_days)

        if saved_lead_days != int(lead_days):
            raise ValueError(
                "lead_days does not match the checkpoint.\n"
                f"Checkpoint lead_days: {saved_lead_days}\n"
                f"Current lead_days:    {lead_days}"
            )

    if "input_mode" in checkpoint:
        saved_input_mode = checkpoint["input_mode"]
        if saved_input_mode != current_input_mode:
            raise ValueError(
                "input_mode does not match the checkpoint.\n"
                f"Checkpoint input_mode: {saved_input_mode}\n"
                f"Current input_mode:    {current_input_mode}"
            )
    elif int(lead_days) > 0:
        raise ValueError(
            "The checkpoint does not contain input_mode metadata and was trained "
            "with the legacy leading-time definition. It cannot be used with "
            "antecedent_mean inputs when lead_days > 0; retrain the model."
        )
    else:
        logging.warning(
            "Checkpoint has no input_mode metadata. Continuing because lead_days=0 "
            "has the same input definition in the legacy and antecedent-mean setups."
        )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    return checkpoint


def calculate_metrics(
    total_squared_error,
    total_absolute_error,
    total_elements,
    all_preds,
    all_targets,
):
    """
    计算整个验证集的 MSE、MAE 和 R2。
    """
    mse = total_squared_error / total_elements
    mae = total_absolute_error / total_elements

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    ss_res = torch.sum((targets - preds) ** 2).item()

    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2).item()

    if ss_tot == 0.0:
        r2 = float("nan")
    else:
        r2 = 1.0 - ss_res / ss_tot

    return mse, mae, r2


def test_model(
    model,
    data_loader,
    device,
    output_dir,
):
    """
    在验证集上进行推理，并按照 target_date 保存预测结果。
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    all_preds = []
    all_targets = []

    saved_count = 0

    progress_bar = tqdm(
        enumerate(data_loader, start=1),
        total=len(data_loader),
        desc="Testing",
        dynamic_ncols=True,
        leave=True,
    )

    with torch.no_grad():
        for iteration, batch in progress_bar:
            (indices,inputs,targets,input_start_dates,input_end_dates,target_dates,) = batch

            inputs = inputs.to(device,non_blocking=True,)
            targets = targets.to(device,non_blocking=True,)

            predictions = model(inputs)

            if predictions.shape != targets.shape:
                raise RuntimeError(
                    "Prediction shape does not match target shape.\n"
                    f"Prediction shape: {tuple(predictions.shape)}\n"
                    f"Target shape:     {tuple(targets.shape)}"
                )

            difference = predictions - targets

            total_squared_error += torch.sum(difference ** 2).item()
            total_absolute_error += torch.sum(torch.abs(difference)).item()
            total_elements += targets.numel()

            all_preds.append(predictions.detach().cpu().reshape(-1))
            all_targets.append(targets.detach().cpu().reshape(-1))

            # 转到 CPU，并保存为 [H, W]
            predictions_numpy = (predictions.detach().cpu().numpy().astype(np.float32))

            batch_size = predictions_numpy.shape[0]

            for batch_index in range(batch_size):
                target_date = str(target_dates[batch_index])

                prediction_array = predictions_numpy[batch_index,0,:,:,]

                save_path = os.path.join(output_dir,f"{target_date}.npy",)
                np.save(save_path,prediction_array,)
                saved_count += 1

            running_mse = (total_squared_error / total_elements)
            running_mae = (total_absolute_error / total_elements)

            progress_bar.set_postfix(
                iteration=f"{iteration}/{len(data_loader)}",
                saved=saved_count,
                mse=f"{running_mse:.6f}",
                mae=f"{running_mae:.6f}",
                input_period=(
                    f"{input_start_dates[-1]}~{input_end_dates[-1]}"
                ),
                target_date=str(target_dates[-1]),
            )

    mse, mae, r2 = calculate_metrics(
        total_squared_error=total_squared_error,
        total_absolute_error=total_absolute_error,
        total_elements=total_elements,
        all_preds=all_preds,
        all_targets=all_targets,
    )

    return mse, mae, r2, saved_count


def main():
    for config_path in config_path_list:
        config = load_config(config_path)

        experiment_name = config["experiment"]["name"]
        seed = int(config["experiment"]["seed"])

        valid_root = config["data"]["valid_root"]
        lead_days = int(config["data"]["lead_days"])
        input_mode = config["data"].get("input_mode", "antecedent_mean")

        shape_scale = (
            int(config["data"]["height"]),
            int(config["data"]["width"]),
        )

        checkpoint_path = config["test"]["checkpoint_path"]
        output_dir = config["test"]["output_dir"]
        log_dir = config["test"]["log_dir"]

        batch_size = int(config["test"]["batch_size"])
        num_workers = int(config["test"]["num_workers"])

        set_seed(seed)
        log_path = setup_logging(log_dir)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 只创建 valid 数据集，不读取 train_root
        valid_dataset = MISOReconstructionDataset(
            root=valid_root,
            variables_config=config["data"]["variables"],
            target_config=config["data"]["target"],
            lead_days=lead_days,
            shape_scale=shape_scale,
            input_mode=input_mode,
        )

        valid_loader = build_dataloader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        model = DilatedSEReconstructionNet(
            in_channels=valid_dataset.num_input_channels,
            stem_channels=int(
                config["model"]["stem_channels"]
            ),
            block_channels=config["model"]["block_channels"],
            dilations=config["model"]["dilations"],
            se_reduction=int(
                config["model"]["se_reduction"]
            ),
        ).to(device)

        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
            channel_names=valid_dataset.channel_names,
            lead_days=lead_days,
            current_input_mode=input_mode,
        )

        logging.info("=" * 60)
        logging.info("Experiment name: %s", experiment_name)
        logging.info("Log path: %s", log_path)
        logging.info("Device: %s", device)
        logging.info("Valid root: %s", valid_root)
        logging.info("Checkpoint: %s", checkpoint_path)
        logging.info("Output directory: %s", output_dir)
        logging.info("Input mode: %s", input_mode)
        logging.info("Lead days: %d", lead_days)
        logging.info("Valid samples: %d", len(valid_dataset))
        logging.info(
            "Input channels total: %d",
            valid_dataset.num_input_channels,
        )
        logging.info(
            "Model parameters: %d",
            model.count_parameters(),
        )

        for channel_index, channel_name in enumerate(
            valid_dataset.channel_names
        ):
            logging.info(
                "Channel %d: %s",
                channel_index,
                channel_name,
            )

        if "epoch" in checkpoint:
            logging.info(
                "Loaded checkpoint epoch: %s",
                checkpoint["epoch"],
            )

        if "valid_mse" in checkpoint:
            logging.info(
                "Checkpoint valid MSE: %s",
                checkpoint["valid_mse"],
            )

        logging.info(
            "Prediction files are named using target_date."
        )
        if lead_days == 0:
            logging.info("Input definition: X(t) -> TP(t).")
        else:
            logging.info(
                "Input definition: mean[X(t-%d), ..., X(t-1)] -> TP(t).",
                lead_days,
            )
        logging.info(
            "Each enabled variable is averaged independently over the antecedent window."
        )
        logging.info("=" * 60)

        test_mse, test_mae, test_r2, saved_count = test_model(
            model=model,
            data_loader=valid_loader,
            device=device,
            output_dir=output_dir,
        )

        logging.info("=" * 60)
        logging.info("Test completed.")
        logging.info("Saved prediction files: %d", saved_count)
        logging.info("Test MSE: %.8f", test_mse)
        logging.info("Test MAE: %.8f", test_mae)

        if np.isnan(test_r2):
            logging.info("Test R2: nan")
        else:
            logging.info("Test R2: %.8f", test_r2)

        logging.info(
            "Prediction output directory: %s",
            output_dir,
        )
        logging.info("=" * 60)


if __name__ == "__main__":
    main()
