import numpy as np

from src.scenario.processes.BrownianMotionGenerator import BrownianMotionGenerator


def test_bm_generator():

    bmg = BrownianMotionGenerator()
    paths_a = bmg.generate(np.zeros((10, 1)), np.arange(5))
    paths_b = bmg.generate(np.zeros((10, 1)), np.arange(5), None, np.random.default_rng(199))
    paths_c = bmg.generate(np.zeros((10, 1)), np.arange(5), np.diff(paths_a, 1, 1))

    assert paths_a.shape == (10, 5, 1,)
    assert paths_b.shape == (10, 5, 1,)
    assert np.all(paths_c == paths_a)
    assert np.any(paths_b != paths_a)
