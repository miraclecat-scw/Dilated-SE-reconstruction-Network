"""
    @Author: Su changwei
    @Email: scw727@outlook.com
    @Date: 2026-07-26
    @Description:
    Use Gradient SHAP to explain the spatial-mean prediction of
    DilatedSEReconstructionNet.

    Model input:
        [B, 15, 31, 21]

    Model output:
        [B, 1, 31, 21]

    Explained scalar:
        mean prediction over the complete 31 x 21 field

    Saved SHAP array:
        [15, 31, 21]
"""

import logging
import os
import random
from datetime import datetime

import numpy as np
import shap
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import MISOReconstructionDataset
from model import DilatedSEReconstructionNet


# 可以一次运行一个或多个 YAML 配置
config_path_list = [
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading0.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading2.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading4.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading6.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading8.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading10.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading12.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading14.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading16.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading18.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading20.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading22.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading24.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading26.yaml",
    "C:/Users/sysu/Desktop/tp_reconstruction/config/leading28.yaml",]


def load_config(config_path):
    """读取 YAML 配置文件。"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def set_seed(seed):
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir):
    """日志同时保存到文件并输出到终端。"""
    os.makedirs(log_dir, exist_ok=True)

    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(
        log_dir,
        f"gradient_shap_{time_string}.log",
    )

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


def build_dataloader(
    dataset,
    batch_size,
    num_workers,
    device,
):
    """只为 valid 数据集建立 DataLoader。"""
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


class SpatialMeanWrappedModel(torch.nn.Module):
    """
    包装原模型，将二维预测场转换成每个样本的一个标量。

    输入：
        [B, 15, 31, 21]

    原模型输出：
        [B, 1, 31, 21]

    包装模型输出：
        [B, 1]

    输出标量是整个 31×21 预测场的空间平均值。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        predictions = self.model(inputs)

        # 兼容模型返回 tuple 的情况
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        if predictions.ndim != 4:
            raise RuntimeError(
                "Expected model output shape [B, 1, H, W], "
                f"but received {tuple(predictions.shape)}"
            )

        if predictions.shape[1] != 1:
            raise RuntimeError(
                "Expected one output channel, "
                f"but received {predictions.shape[1]} channels."
            )

        # 只对 H、W 两个空间维度取平均
        # [B, 1, 31, 21] -> [B, 1]
        mean_predictions = predictions.mean(
            dim=(-2, -1)
        )

        return mean_predictions


def load_checkpoint(
    checkpoint_path,
    model,
    device,
    current_channel_names,
    current_lead_days,
    current_input_mode,
):
    """读取 .tar checkpoint，并加载模型参数。"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False,)
    except TypeError:
        checkpoint = torch.load(checkpoint_path,map_location=device,)

    # 兼容两种保存方式：
    # 1. checkpoint 字典，参数位于 state_dict
    # 2. 直接保存的 model.state_dict()
    if (isinstance(checkpoint, dict)and "state_dict" in checkpoint):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict,strict=True,)

    # 检查输入通道顺序
    if (isinstance(checkpoint, dict) and "channel_names" in checkpoint):
        checkpoint_channel_names = list(checkpoint["channel_names"])
        current_channel_names = list(current_channel_names)

        if checkpoint_channel_names != current_channel_names:
            raise ValueError(
                "The channel order in the YAML does not match "
                "the channel order used during training.\n"
                f"Checkpoint channels: {checkpoint_channel_names}\n"
                f"Current channels:    {current_channel_names}"
            )

    # 检查 lead_days
    if (isinstance(checkpoint, dict) and "lead_days" in checkpoint):
        checkpoint_lead_days = int(checkpoint["lead_days"])

        if checkpoint_lead_days != int(current_lead_days):
            raise ValueError(
                "The lead_days in the YAML does not match "
                "the checkpoint.\n"
                f"Checkpoint lead_days: {checkpoint_lead_days}\n"
                f"Current lead_days:    {current_lead_days}"
            )

    if isinstance(checkpoint, dict) and "input_mode" in checkpoint:
        saved_input_mode = checkpoint["input_mode"]
        if saved_input_mode != current_input_mode:
            raise ValueError(
                "The input_mode in the YAML does not match the checkpoint.\n"
                f"Checkpoint input_mode: {saved_input_mode}\n"
                f"Current input_mode:    {current_input_mode}"
            )
    elif int(current_lead_days) > 0:
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

    return checkpoint


def convert_shap_values(
    shap_values,
    expected_input_shape,
):
    """
    将不同 SHAP 版本的输出统一转换为：

        [B, C, H, W]

    对于单输出模型，部分 SHAP 版本返回：

        [B, C, H, W]

    新版本也可能返回：

        [B, C, H, W, 1]
    """
    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise RuntimeError(
                "The wrapped model should have only one output, "
                f"but SHAP returned {len(shap_values)} outputs."
            )

        shap_values = shap_values[0]

    shap_array = np.asarray(
        shap_values,
        dtype=np.float32,
    )

    expected_input_shape = tuple(
        expected_input_shape
    )

    # 已经是 [B, C, H, W]
    if shap_array.shape == expected_input_shape:
        return shap_array

    # SHAP 新版本可能是 [B, C, H, W, 1]
    if shap_array.shape == expected_input_shape + (1,):
        return shap_array[..., 0]

    raise RuntimeError(
        "Unexpected SHAP output shape.\n"
        f"Expected: {expected_input_shape} "
        f"or {expected_input_shape + (1,)}\n"
        f"Received: {shap_array.shape}"
    )


def run_gradient_shap(
    model,
    data_loader,
    device,
    save_dir,
    number_of_channels,
    height,
    width,
    nsamples,
    explainer_batch_size,
):
    """
    对 valid 数据集逐批计算 Gradient SHAP。

    每个样本保存一个：

        target_date.npy

    每个文件形状为：

        [15, 31, 21]
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    wrapped_model = SpatialMeanWrappedModel(
        model
    ).to(device)
    wrapped_model.eval()

    # baseline 始终为 0
    # 只使用一个全零背景样本
    baseline_inputs = torch.zeros(
        (
            1,
            number_of_channels,
            height,
            width,
        ),
        dtype=torch.float32,
        device=device,
    )

    explainer = shap.GradientExplainer(
        wrapped_model,
        baseline_inputs,
        batch_size=explainer_batch_size,
        local_smoothing=0,
    )

    saved_count = 0

    progress_bar = tqdm(
        enumerate(data_loader, start=1),
        total=len(data_loader),
        desc="Gradient SHAP",
        dynamic_ncols=True,
    )

    # 注意：这里不能使用 torch.no_grad()
    # Gradient SHAP 必须计算输入梯度
    for iteration, batch_data in progress_bar:
        (
            indices,
            inputs,
            targets,
            input_start_dates,
            input_end_dates,
            target_dates,
        ) = batch_data

        inputs = inputs.to(device,non_blocking=True,)

        expected_shape = (inputs.shape[0],number_of_channels,height,width,)

        if tuple(inputs.shape) != expected_shape:
            raise RuntimeError(
                "Invalid model input shape.\n"
                f"Expected: {expected_shape}\n"
                f"Received: {tuple(inputs.shape)}"
            )

        # 计算输入变量对“整张预测场平均值”的贡献
        shap_values = explainer.shap_values(inputs,nsamples=nsamples,ranked_outputs=None,)
        shap_array = convert_shap_values(shap_values=shap_values,expected_input_shape=expected_shape,)

        # shap_array:
        # [B, 15, 31, 21]
        for batch_index in range(shap_array.shape[0]):
            target_date = str(target_dates[batch_index])

            sample_shap = shap_array[batch_index].astype(np.float32)
            expected_sample_shape = (number_of_channels,height,width,)

            if sample_shap.shape != expected_sample_shape:
                raise RuntimeError(
                    "Invalid saved SHAP shape.\n"
                    f"Expected: {expected_sample_shape}\n"
                    f"Received: {sample_shap.shape}"
                )

            # 使用 target 日期命名
            # 表示这些输入贡献对应的是该日期的预测 TP
            save_path = os.path.join(save_dir,f"{target_date}.npy",)

            if os.path.exists(save_path):
                raise FileExistsError(
                    "A SHAP file with the same target date "
                    f"already exists: {save_path}"
                )

            np.save(save_path,sample_shap,)
            saved_count += 1

        progress_bar.set_postfix(
            iteration=f"{iteration}/{len(data_loader)}",
            saved=saved_count,
            input_period=(
                f"{input_start_dates[-1]}~"
                f"{input_end_dates[-1]}"
            ),
            target_date=str(target_dates[-1]),)

    return saved_count


def main():
    for config_path in config_path_list:
        config = load_config(config_path)

        experiment_name = config["experiment"]["name"]
        seed = int(config["experiment"]["seed"])

        valid_root = config["data"]["valid_root"]
        lead_days = int(config["data"]["lead_days"])
        input_mode = config["data"].get("input_mode", "antecedent_mean")

        height = int(config["data"]["height"])
        width = int(config["data"]["width"])

        shap_config = config["gradient_shap"]

        checkpoint_path = shap_config["checkpoint_path"]
        shap_save_dir = shap_config["save_dir"]
        shap_log_dir = shap_config.get("log_dir",shap_save_dir,)

        batch_size = int(shap_config.get("batch_size", 1))
        num_workers = int(shap_config.get("num_workers", 0))
        nsamples = int(shap_config.get("nsamples", 200))
        explainer_batch_size = int(shap_config.get("explainer_batch_size",16,))

        os.makedirs(shap_save_dir,exist_ok=True,)

        log_path = setup_logging(shap_log_dir)
        set_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 只读取 valid 数据集
        valid_dataset = MISOReconstructionDataset(
            root=valid_root,
            variables_config=config["data"]["variables"],
            target_config=config["data"]["target"],
            lead_days=lead_days,
            shape_scale=(height, width),
            input_mode=input_mode,
        )

        # 本次任务明确要求 15 个通道
        if valid_dataset.num_input_channels != 15:
            raise ValueError(
                "This Gradient SHAP program expects exactly "
                "15 input channels, but the dataset contains "
                f"{valid_dataset.num_input_channels} channels."
            )

        valid_loader = build_dataloader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        model = DilatedSEReconstructionNet(
            in_channels=(valid_dataset.num_input_channels),
            stem_channels=int(config["model"]["stem_channels"]),
            block_channels=(config["model"]["block_channels"]),
            dilations=(config["model"]["dilations"]),
            se_reduction=int(config["model"]["se_reduction"]),).to(device)

        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
            current_channel_names=(valid_dataset.channel_names),
            current_lead_days=lead_days,
            current_input_mode=input_mode,)

        logging.info("=" * 70)
        logging.info("Experiment name: %s",experiment_name,)
        logging.info("Configuration file: %s",config_path,)
        logging.info("Log file: %s", log_path,)
        logging.info("Device: %s",device,)
        logging.info("Valid root: %s",valid_root,)
        logging.info("Checkpoint: %s",checkpoint_path,)
        logging.info("SHAP save directory: %s",shap_save_dir,)
        logging.info("Valid samples: %d",len(valid_dataset),)
        logging.info("Input shape: [B, %d, %d, %d]",valid_dataset.num_input_channels,height, width,)
        logging.info("Baseline: all zeros")
        logging.info("Gradient SHAP nsamples: %d", nsamples,)
        logging.info(
            "Explained output: spatial mean "
            "of the complete %d x %d prediction field",
            height,width,)
        logging.info("Lead days: %d",lead_days,)
        logging.info("Input mode: %s",input_mode,)
        if lead_days == 0:
            logging.info("Gradient SHAP input: X(t).")
        else:
            logging.info(
                "Gradient SHAP input: per-variable mean over X(t-%d) ... X(t-1).",
                lead_days,
            )

        for channel_index, channel_name in enumerate(
            valid_dataset.channel_names
        ):
            logging.info("Channel %d: %s",channel_index,channel_name,)

        if (isinstance(checkpoint, dict) and "epoch" in checkpoint):
            logging.info("Checkpoint epoch: %s",checkpoint["epoch"],)

        logging.info("=" * 70)

        saved_count = run_gradient_shap(
            model=model,
            data_loader=valid_loader,
            device=device,
            save_dir=shap_save_dir,
            number_of_channels=(
                valid_dataset.num_input_channels
            ),
            height=height,
            width=width,
            nsamples=nsamples,
            explainer_batch_size=(
                explainer_batch_size
            ),
        )

        logging.info("=" * 70)
        logging.info("Gradient SHAP completed.")
        logging.info("Saved SHAP files: %d",saved_count,)
        logging.info(
            "Each SHAP file has shape: "
            "[%d, %d, %d]",
            valid_dataset.num_input_channels,
            height,
            width,
        )
        logging.info("Files are named using target dates.")
        logging.info("Output directory: %s",shap_save_dir,)
        logging.info("=" * 70)


if __name__ == "__main__":
    main()
