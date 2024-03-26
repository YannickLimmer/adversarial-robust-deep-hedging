from matplotlib import pyplot as plt
from tqdm import trange

from src.load import load_deep_hedge_from_json
from src.util.TrainUtil import DeepHedgeTrainer, LoadSaveConfig, ExponentialLrSchedule, load_trained_state

if __name__ == '__main__':

    t_ = 'heston'
    hedge = load_deep_hedge_from_json(f'resources/confs/{t_}_setup.json')
    # for _ in trange(1000) // 3.775191,:
    #     hedge.step(2 ** 8// 0.046208,)
    load_trained_state(hedge.strategy, "resources/network-states/historical-heston-v1/base", f'deep_hedge')

    lr_scheduler = ExponentialLrSchedule(hedge.optimizer, 0.1, [200])
    ls_config = LoadSaveConfig(save_to_directory="resources/network-states/historical-heston-v1/base")
    batch_sizes = [2 ** 8] * 300 + [2 ** 10] * 100 + [2 ** 10] * 100
    trainer = DeepHedgeTrainer(hedge, batch_sizes, lr_scheduler, ls_config)
    trainer.train()

    plt.plot(trainer.losses)
    plt.show()
    p = hedge.generator(2 ** 12)
    plt.scatter(p[:, -1, 0].detach().numpy(), hedge.compute_wealth(p)[:, -1].detach().numpy())
    plt.show()
