"""
    @Author: Su changwei
    @Email: scw727@outlook.com
    @Date: 2026-07-26
    @Description: Dataset for TP 2D field reconstruction.
    @File: dataset.py
"""

import os
import re
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset


class MISOReconstructionDataset(Dataset):
    def __init__(
        self,
        root,
        variables_config,
        target_config,
        lead_days,
        shape_scale=(31, 21),
        input_mode="antecedent_mean",
    ):
        self.root = root
        self.variables_config = variables_config
        self.target_config = target_config
        self.lead_days = int(lead_days)
        self.shape_scale = tuple(shape_scale)
        self.input_mode = input_mode

        if self.input_mode != "antecedent_mean":
            raise ValueError(
                "Unsupported input_mode: "
                f"{self.input_mode!r}. Only 'antecedent_mean' is supported."
            )
        if self.lead_days < 0:
            raise ValueError(
                f"lead_days must be greater than or equal to 0, got {self.lead_days}."
            )

        self.channel_names = [
            name for name, cfg in self.variables_config.items() if bool(cfg.get("enabled", False))
        ]
        self.num_input_channels = len(self.channel_names)

        if self.num_input_channels == 0:
            raise ValueError("At least one input variable must be enabled in config.yaml.")

        self.sample_list = self._build_sample_list()

        if not self.sample_list:
            raise RuntimeError(f"No valid samples found under dataset root: {self.root}")

        self._print_summary()

    def _build_sample_list(self):
        target_folder = self.target_config["folder"]
        target_folder_path = os.path.join(self.root, target_folder)

        if not os.path.isdir(target_folder_path):
            raise RuntimeError(f"Target folder does not exist: {target_folder_path}")

        filename_pattern = re.compile(r"^\d{8}\.npy$")
        candidate_files = sorted(
            file_name
            for file_name in os.listdir(target_folder_path)
            if filename_pattern.match(file_name)
        )

        sample_list = []
        expected_input_file_count = self.lead_days if self.lead_days > 0 else 1

        for file_name in candidate_files:
            target_date = os.path.splitext(file_name)[0]
            try:
                input_dates = self._get_input_dates(target_date)
            except ValueError:
                continue

            if len(input_dates) != expected_input_file_count:
                raise RuntimeError(
                    "Unexpected input date count for target_date="
                    f"{target_date}: expected {expected_input_file_count}, "
                    f"got {len(input_dates)}."
                )

            target_path = os.path.join(target_folder_path, file_name)
            if not os.path.isfile(target_path) or not self._has_expected_shape(target_path):
                continue

            input_paths = {}
            is_valid = True

            for channel_name in self.channel_names:
                folder_name = self.variables_config[channel_name]["folder"]
                channel_paths = [
                    os.path.join(self.root, folder_name, f"{input_date}.npy")
                    for input_date in input_dates
                ]

                if len(channel_paths) != expected_input_file_count:
                    raise RuntimeError(
                        "Unexpected input path count for variable="
                        f"{channel_name}, target_date={target_date}: "
                        f"expected {expected_input_file_count}, got {len(channel_paths)}."
                    )

                for input_path in channel_paths:
                    if not os.path.isfile(input_path) or not self._has_expected_shape(input_path):
                        is_valid = False
                        break

                if not is_valid:
                    break

                input_paths[channel_name] = channel_paths

            if not is_valid:
                continue

            sample_list.append(
                {
                    "target_date": target_date,
                    "input_dates": input_dates,
                    "input_start_date": input_dates[0],
                    "input_end_date": input_dates[-1],
                    "input_paths": input_paths,
                    "target_path": target_path,
                }
            )

        return sample_list

    def _get_input_dates(self, target_date):
        target_datetime = datetime.strptime(target_date, "%Y%m%d")

        if self.lead_days == 0:
            return [target_date]

        return [
            (target_datetime - timedelta(days=offset)).strftime("%Y%m%d")
            for offset in range(self.lead_days, 0, -1)
        ]

    def _has_expected_shape(self, file_path):
        try:
            array = np.load(file_path, mmap_mode="r", allow_pickle=False)
        except Exception:
            return False
        return array.ndim == 2 and tuple(array.shape) == self.shape_scale

    def _validate_array(self, array, variable_name, date_string, file_path):
        if array.ndim != 2 or tuple(array.shape) != self.shape_scale:
            raise ValueError(
                f"Invalid shape for variable={variable_name}, date={date_string}, "
                f"path={file_path}, shape={tuple(array.shape)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"NaN or Inf found in variable={variable_name}, date={date_string}, path={file_path}"
            )

    def _print_summary(self):
        first_sample = self.sample_list[0]
        last_sample = self.sample_list[-1]

        if self.lead_days == 0:
            input_definition = "X(t) -> TP(t)"
        else:
            input_definition = (
                f"mean[X(t-{self.lead_days}), ..., X(t-1)] -> TP(t)"
            )

        print("=" * 60)
        print(f"Dataset root: {self.root}")
        print(f"Valid samples: {len(self.sample_list)}")
        print(f"Input mode: {self.input_mode}")
        print(f"Lead days: {self.lead_days}")
        print("Input definition:")
        print(input_definition)
        print("Input channels:")
        for index, channel_name in enumerate(self.channel_names):
            print(f"  Channel {index}: {channel_name}")
        print(f"Number of input channels: {self.num_input_channels}")
        print(
            "First sample: "
            f"input_period={first_sample['input_start_date']} to "
            f"{first_sample['input_end_date']}, "
            f"target_date={first_sample['target_date']}"
        )
        print(
            "Last sample: "
            f"input_period={last_sample['input_start_date']} to "
            f"{last_sample['input_end_date']}, "
            f"target_date={last_sample['target_date']}"
        )
        print("=" * 60)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.sample_list[index]

        inputs_list = []
        for channel_name in self.channel_names:
            channel_paths = sample["input_paths"][channel_name]

            if len(channel_paths) != len(sample["input_dates"]):
                raise RuntimeError(
                    "Input path count does not match input date count for variable="
                    f"{channel_name}, target_date={sample['target_date']}."
                )

            variable_arrays = []
            for input_date, input_path in zip(sample["input_dates"], channel_paths):
                input_array = np.load(input_path, allow_pickle=False).astype(np.float32)
                self._validate_array(input_array, channel_name, input_date, input_path)
                variable_arrays.append(input_array)

            variable_stack = np.stack(variable_arrays, axis=0)
            variable_mean = np.mean(variable_stack, axis=0, dtype=np.float32)
            inputs_list.append(variable_mean)

        target_array = np.load(sample["target_path"], allow_pickle=False).astype(np.float32)
        self._validate_array(
            target_array,
            self.target_config["name"],
            sample["target_date"],
            sample["target_path"],
        )

        inputs = np.stack(inputs_list, axis=0)
        target = np.expand_dims(target_array, axis=0)

        expected_inputs_shape = (self.num_input_channels, *self.shape_scale)
        expected_target_shape = (1, *self.shape_scale)
        if tuple(inputs.shape) != expected_inputs_shape:
            raise RuntimeError(
                f"Invalid final input shape: expected {expected_inputs_shape}, got {tuple(inputs.shape)}."
            )
        if tuple(target.shape) != expected_target_shape:
            raise RuntimeError(
                f"Invalid final target shape: expected {expected_target_shape}, got {tuple(target.shape)}."
            )

        inputs = torch.from_numpy(inputs).float()
        target = torch.from_numpy(target).float()

        return (
            index,
            inputs,
            target,
            sample["input_start_date"],
            sample["input_end_date"],
            sample["target_date"],
        )
