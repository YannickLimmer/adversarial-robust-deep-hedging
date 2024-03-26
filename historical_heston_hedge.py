from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.config import DEVICE
from src.load import load_deep_hedge_from_json
from src.util.TrainUtil import DeepHedgeTrainer, LoadSaveConfig, ExponentialLrSchedule, load_trained_state

import ray
import torch

from src.util.ray_util.TqdmUtil import to_limited_iterator

ray.init()


@ray.remote
def train_trainer(d, t: DeepHedgeTrainer, pars):
    t = deepcopy(t)
    t.deep_hedge.generator.reversion_speed.val = torch.tensor(np.abs(pars.reversion_speed), dtype=torch.float32, device=DEVICE)
    t.deep_hedge.generator.reversion_level.val = torch.tensor(np.abs(pars.reversion_level), dtype=torch.float32, device=DEVICE)
    t.deep_hedge.generator.vol_of_vol.val = torch.tensor(np.abs(pars.vol_of_vol), dtype=torch.float32, device=DEVICE)
    t.deep_hedge.generator.correlation.val = torch.tensor(pars.correlation, dtype=torch.float32, device=DEVICE)
    t.ls_config.save_to_directory += f'/{d}'[:11]
    t.train()


if __name__ == '__main__':

    t_ = 'heston'
    hedge = load_deep_hedge_from_json(f'resources/confs/{t_}_setup.json')
    load_trained_state(hedge.strategy, "resources/network-states/historical-heston-v1/base", f'deep_hedge')
    hedge = deepcopy(hedge)
    pars_df = pd.read_pickle("data/optionmetrics/SPX_KF.pkl")
    # for _ in trange(1000) // 3.775191,:
    #     hedge.step(2 ** 8// 0.046208,)

    lr_scheduler = ExponentialLrSchedule(hedge.optimizer, 0.1, [150, 100, 100, 100])
    ls_config = LoadSaveConfig(save_to_directory="resources/network-states/historical-heston-v1")
    batch_sizes = [2 ** 8] * 250 + [2 ** 10] * 200 + [2 ** 12] * 50
    trainer = DeepHedgeTrainer(hedge, batch_sizes, lr_scheduler, ls_config, tqdm=False)

    tasks = (train_trainer.remote(d=d, t=trainer, pars=ps) for d, ps in tqdm(pars_df.iterrows(), total=len(pars_df)))

    for _ in range(5):
        next(to_limited_iterator(tasks, 5))
