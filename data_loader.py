import config

import numpy as np

import torch
from torch.utils.data import DataLoader , Dataset


class GenerateDataLoader:
    def __init__(self, gap, batch_size , train_ratio=config.TRAIN_RATIO, valid_ratio=config.VAL_RATIO ):
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.gap = gap
        self.batch_size = batch_size
        self.num_workers = 0

    def _split_dfs(self, dfs_dict):
        train_dfs, valid_dfs, test_dfs = {}, {}, {}

        for ticker, df in dfs_dict.items():
            n = len(df)
            train_end = int(n * self.train_ratio)
            valid_start = train_end + self.gap
            valid_end = int(n * (self.train_ratio + self.valid_ratio))
            test_start = valid_end + self.gap

            train_dfs[ticker] = df.iloc[:train_end]

            if valid_start < valid_end:
                valid_dfs[ticker] = df.iloc[valid_start:valid_end]

            if test_start < n:
                test_dfs[ticker] = df.iloc[test_start:]

        return train_dfs, valid_dfs, test_dfs

    def __call__(self, dataset_class, dfs_dict, **dataset_kwargs):

        train_dfs, valid_dfs, test_dfs = self._split_dfs(dfs_dict)

        train_ds = dataset_class(train_dfs, **dataset_kwargs)
        valid_ds = dataset_class(valid_dfs, **dataset_kwargs)
        test_ds = dataset_class(test_dfs, **dataset_kwargs)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True)

        return train_loader, valid_loader, test_loader
    
class TimeSeriesDataset(Dataset):
    def __init__(self, dfs_dict, window_size):
        self.window_size = window_size
        self.data_dict = {}
        self.index_map = []

        for ticker, df in dfs_dict.items():
            data = df.values
            if len(data) < window_size:
                continue

            self.data_dict[ticker] = torch.tensor(data, dtype=torch.float32)

            num_windows = len(data) - window_size + 1
            for i in range(num_windows):
                self.index_map.append((ticker, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ticker, start_idx = self.index_map[idx]

        window = self.data_dict[ticker][start_idx : start_idx + self.window_size]
        return window
    
class LabeledTimeSeriesDataset(Dataset):
    def __init__(self, dfs_dict, window_size):
        self.window_size = window_size
        self.data_dict = {}
        self.index_map = []

        for ticker, df in dfs_dict.items():
            data = df.values
            if len(data) < window_size:
                continue

            self.data_dict[ticker] = torch.tensor(data, dtype=torch.float32)
            num_windows = len(data) - window_size + 1
            for i in range(num_windows):
                self.index_map.append((ticker, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ticker, start_idx = self.index_map[idx]

        window = self.data_dict[ticker][start_idx : start_idx + self.window_size]
        x = window[:, :-1]
        y = window[-1, -1]
        return x, y
    

class TNCDataset(Dataset):
    def __init__(self, dfs_dict, window_size, mc_sample_size=10):
        """
        :param mc_sample_size: 이웃(Neighborhood)으로 간주할 윈도우의 범위 (표준편차)
        """
        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.data_dict = {}
        self.index_map = []
        self.ticker_lengths = {}

        for ticker, df in dfs_dict.items():
            data = df.values
            if len(data) < window_size:
                continue

            self.data_dict[ticker] = torch.tensor(data, dtype=torch.float32)
            num_windows = len(data) - window_size + 1
            self.ticker_lengths[ticker] = num_windows
            
            for i in range(num_windows):
                self.index_map.append((ticker, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ticker, t = self.index_map[idx]
        max_t = self.ticker_lengths[ticker] - 1

        # 1. Anchor (현재 시점의 윈도우)
        anchor = self.data_dict[ticker][t : t + self.window_size]

        # 2. Positive (가까운 이웃: 정규분포를 사용해 근처 시점 샘플링)
        p_dist = int(np.random.normal(0, self.mc_sample_size))
        p_t = np.clip(t + p_dist, 0, max_t)
        positive = self.data_dict[ticker][p_t : p_t + self.window_size]

        # 3. Negative (먼 이웃: 무작위 시점 샘플링하되, 이웃 범위를 벗어나도록 강제)
        n_t = np.random.randint(0, max_t + 1)
        while abs(n_t - t) <= self.mc_sample_size * 2: # 이웃 범위를 벗어날 때까지 반복
            n_t = np.random.randint(0, max_t + 1)
            
        negative = self.data_dict[ticker][n_t : n_t + self.window_size]

        return anchor, positive, negative