"""
    @Author: Su changwei
    @Email: scw727@outlook.com
    @Date: 2026-07-26
    @Description: Training script for TP 2D field reconstruction.
    @File: main.py
"""

import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MISOReconstructionDataset
from early_stopping import EarlyStopping
from model import DilatedSEReconstructionNet


try:
    import yaml
except ImportError:
    yaml = None


config_path_list = ["C:/Users/sysu/Desktop/tp_reconstruction/config/leading0.yaml"]
# config_path_list = [
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading0.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading2.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading4.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading6.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading8.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading10.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading12.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading14.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading16.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading18.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading20.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading22.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading24.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading26.yaml",
#     "C:/Users/sysu/Desktop/tp_reconstruction/config/leading28.yaml",]


def _parse_scalar(value):
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]

    lower_value = value.lower()
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _prepare_yaml_lines(text):
    parsed_lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        parsed_lines.append((indent, stripped))
    return parsed_lines


def _parse_yaml_block(lines, start_index, indent):
    current_indent, current_content = lines[start_index]

    if current_indent != indent:
        raise ValueError("Invalid YAML indentation.")

    if current_content.startswith("- "):
        items = []
        index = start_index
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent:
                break
            if line_indent != indent or not content.startswith("- "):
                break

            value_text = content[2:].strip()
            index += 1

            if value_text:
                items.append(_parse_scalar(value_text))
            elif index < len(lines) and lines[index][0] > line_indent:
                child_value, index = _parse_yaml_block(lines, index, lines[index][0])
                items.append(child_value)
            else:
                items.append(None)

        return items, index

    mapping = {}
    index = start_index

    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or content.startswith("- "):
            break

        key, separator, value_text = content.partition(":")
        if separator != ":":
            raise ValueError(f"Invalid YAML line: {content}")

        key = key.strip()
        value_text = value_text.strip()
        index += 1

        if value_text:
            mapping[key] = _parse_scalar(value_text)
        elif index < len(lines) and lines[index][0] > line_indent:
            child_value, index = _parse_yaml_block(lines, index, lines[index][0])
            mapping[key] = child_value
        else:
            mapping[key] = {}

    return mapping, index


def _load_simple_yaml(text):
    lines = _prepare_yaml_lines(text)
    if not lines:
        return {}
    parsed_data, next_index = _parse_yaml_block(lines, 0, lines[0][0])
    if next_index != len(lines):
        raise ValueError("Failed to parse the full YAML file.")
    return parsed_data


def load_config(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return _load_simple_yaml(text)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return log_path


def calculate_metrics(total_squared_error, total_absolute_error, total_elements, all_preds, all_targets):
    mse = total_squared_error / total_elements
    mae = total_absolute_error / total_elements

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    ss_res = torch.sum((targets - preds) ** 2).item()
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2).item()
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)

    return mse, mae, r2


def format_metric(value):
    return "nan" if np.isnan(value) else f"{value:.6f}"


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
):
    model.train()

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    all_preds = []
    all_targets = []

    progress_bar = tqdm(
        enumerate(data_loader, start=1),
        total=len(data_loader),
        desc=f"Train Epoch {epoch}/{total_epochs}",
        dynamic_ncols=True,
        leave=False,
    )

    for iteration, (
        _,
        inputs,
        targets,
        input_start_dates,
        input_end_dates,
        target_dates,
    ) in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        preds = model(inputs)

        # 训练只使用 MSE loss
        mse_loss = criterion(preds, targets)

        mse_loss.backward()
        optimizer.step()

        # 累计整个 epoch 的误差
        diff = preds.detach() - targets

        total_squared_error += torch.sum(diff ** 2).item()
        total_absolute_error += torch.sum(torch.abs(diff)).item()
        total_elements += targets.numel()

        all_preds.append(preds.detach().cpu().reshape(-1))
        all_targets.append(targets.detach().cpu().reshape(-1))

        # 当前 epoch 截止该 iteration 的累计指标
        running_mse = total_squared_error / total_elements
        running_mae = total_absolute_error / total_elements

        # 动态显示当前 iteration 和累计指标
        progress_bar.set_postfix(
            iteration=f"{iteration}/{len(data_loader)}",
            mse=f"{running_mse:.6f}",
            mae=f"{running_mae:.6f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return calculate_metrics(
        total_squared_error=total_squared_error,
        total_absolute_error=total_absolute_error,
        total_elements=total_elements,
        all_preds=all_preds,
        all_targets=all_targets,
    )


def validate_one_epoch(
    model,
    data_loader,
    device,
    epoch,
    total_epochs,
):
    model.eval()

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    all_preds = []
    all_targets = []

    progress_bar = tqdm(
        enumerate(data_loader, start=1),
        total=len(data_loader),
        desc=f"Valid Epoch {epoch}/{total_epochs}",
        dynamic_ncols=True,
        leave=False,
    )

    with torch.no_grad():
        for iteration, (
            _,
            inputs,
            targets,
            input_start_dates,
            input_end_dates,
            target_dates,
        ) in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model(inputs)
            diff = preds - targets

            total_squared_error += torch.sum(diff ** 2).item()
            total_absolute_error += torch.sum(torch.abs(diff)).item()
            total_elements += targets.numel()

            all_preds.append(preds.detach().cpu().reshape(-1))
            all_targets.append(targets.detach().cpu().reshape(-1))

            running_mse = total_squared_error / total_elements
            running_mae = total_absolute_error / total_elements

            progress_bar.set_postfix(iteration=f"{iteration}/{len(data_loader)}",mse=f"{running_mse:.6f}",mae=f"{running_mae:.6f}",)

    return calculate_metrics(
        total_squared_error=total_squared_error,
        total_absolute_error=total_absolute_error,
        total_elements=total_elements,
        all_preds=all_preds,
        all_targets=all_targets,
    )

def build_dataloader(dataset, batch_size, num_workers, shuffle):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(**loader_kwargs)


def main():
    for config_path in config_path_list:
        config = load_config(config_path)

        experiment_name = config["experiment"]["name"]
        seed = int(config["experiment"]["seed"])
        log_dir = config["paths"]["log_dir"]
        save_dir = config["paths"]["save_dir"]
        train_root = config["data"]["train_root"]
        valid_root = config["data"]["valid_root"]
        lead_days = int(config["data"]["lead_days"])
        input_mode = config["data"].get("input_mode", "antecedent_mean")
        shape_scale = (int(config["data"]["height"]), int(config["data"]["width"]))

        os.makedirs(save_dir, exist_ok=True)
        log_file_path = setup_logging(log_dir)
        set_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset = MISOReconstructionDataset(
            root=train_root,
            variables_config=config["data"]["variables"],
            target_config=config["data"]["target"],
            lead_days=lead_days,
            shape_scale=shape_scale,
            input_mode=input_mode,
        )
        valid_dataset = MISOReconstructionDataset(
            root=valid_root,
            variables_config=config["data"]["variables"],
            target_config=config["data"]["target"],
            lead_days=lead_days,
            shape_scale=shape_scale,
            input_mode=input_mode,
        )

        batch_size = int(config["training"]["batch_size"])
        num_workers = int(config["training"]["num_workers"])
        epochs = int(config["training"]["epochs"])

        train_loader = build_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        valid_loader = build_dataloader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        model = DilatedSEReconstructionNet(
            in_channels=train_dataset.num_input_channels,
            stem_channels=int(config["model"]["stem_channels"]),
            block_channels=config["model"]["block_channels"],
            dilations=config["model"]["dilations"],
            se_reduction=int(config["model"]["se_reduction"]),
        ).to(device)

        # criterion = nn.MSELoss()
        criterion = nn.L1Loss() # 使用 L1 loss
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["optimizer"]["lr"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(config["scheduler"]["t_max"]),
            eta_min=float(config["scheduler"]["eta_min"]),
        )

        early_stopping = EarlyStopping(
            patience=int(config["early_stopping"]["patience"]),
            verbose=True,
        )

        logging.info("Experiment name: %s", experiment_name)
        logging.info("Log file: %s", log_file_path)
        logging.info("Device: %s", device)
        logging.info("Train root: %s", train_root)
        logging.info("Valid root: %s", valid_root)
        logging.info("Input mode: %s", input_mode)
        logging.info("Lead days: %d", lead_days)
        logging.info("Train samples: %d", len(train_dataset))
        logging.info("Valid samples: %d", len(valid_dataset))
        logging.info("Input channels total: %d", train_dataset.num_input_channels)
        logging.info("Model parameters: %d", model.count_parameters())
        for channel_index, channel_name in enumerate(train_dataset.channel_names):
            logging.info("Channel %d: %s", channel_index, channel_name)
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

        last_model_state = None

        for epoch in range(1, epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]

            train_mse, train_mae, train_r2 = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                total_epochs=epochs,
            )

            logging.info(
                "Epoch [%d/%d](Train) | MSE=%s | MAE=%s | R2=%s | lr=%.8f",
                epoch,
                epochs,
                format_metric(train_mse),
                format_metric(train_mae),
                format_metric(train_r2),
                current_lr,
            )

            valid_mse = float("nan")
            valid_mae = float("nan")
            valid_r2 = float("nan")

            if epoch % 10 == 0 and epoch != 0:
                valid_mse, valid_mae, valid_r2 = validate_one_epoch(
                    model=model,
                    data_loader=valid_loader,
                    device=device,
                    epoch=epoch,
                    total_epochs=epochs,
                )

                logging.info(
                    "Epoch [%d/%d](Valid) | MSE=%s | MAE=%s | R2=%s | lr=%.8f",
                    epoch,
                    epochs,
                    format_metric(valid_mse),
                    format_metric(valid_mae),
                    format_metric(valid_r2),
                    current_lr,
                )

            scheduler.step()

            model_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "valid_mse": valid_mse,
                "valid_mae": valid_mae,
                "valid_r2": valid_r2,
                "channel_names": train_dataset.channel_names,
                "lead_days": lead_days,
                "input_mode": input_mode,
                "input_definition": (
                    "X(t) -> TP(t)"
                    if lead_days == 0
                    else f"mean[X(t-{lead_days}), ..., X(t-1)] -> TP(t)"
                ),
            }
            last_model_state = model_state

            if epoch % 10 == 0 and epoch != 0:
                early_stopping(
                    valid_mse,
                    model_state,
                    epoch,
                    save_dir,
                )

                if early_stopping.early_stop:
                    logging.info("Early stopping triggered.")
                    break

        torch.save(last_model_state, os.path.join(save_dir, "last_model.tar"))
        logging.info("Saved last model to %s", os.path.join(save_dir, "last_model.tar"))


if __name__ == "__main__":
    main()
