import json

import numpy as np
from torch import nn

from src.deep_hedging.StrategyNet import StrategyNet, StrategyNetConfig
from src.deep_hedging.objectives.HedgeObjective import Std, MeanVariance, StableEntropy, CVAR
from src.dis.DeepHedge import Masks, DeepHedge
from src.dis.Derivative import EuroCall, MaxEuroCall
from src.gen.BlackScholes import BlackScholesParameterSet, BlackScholesGeneratorConfig, BlackScholesGenerator
from src.gen.Heston import HestonParameterSet, HestonGeneratorConfig, HestonGenerator
from src.util.TimeUtil import UniformTimeDiscretization

PARAMETER_SETS = {
    "Heston": HestonParameterSet,
    "BlackScholes": BlackScholesParameterSet,
}
GENERATOR_CONFIGS = {
    "Heston": HestonGeneratorConfig,
    "BlackScholes": BlackScholesGeneratorConfig,
}
GENERATORS = {
    "Heston": HestonGenerator,
    "BlackScholes": BlackScholesGenerator,
}
DERIVATIVES = {
    "EuroCall": EuroCall,
    "MaxEuroCall": MaxEuroCall,
}
OBJECTIVES = {
    "Std": Std,
    "MeanVariance": MeanVariance,
    "Entropy": StableEntropy,
    "CVAR": CVAR,
}


def load_deep_hedge_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        config = json.load(f)

    gc = config["Generator"]

    parameter_set = PARAMETER_SETS[gc['ParameterSetType']].from_json(gc['ParameterSet'])
    td = UniformTimeDiscretization.from_bounds(**gc['TimeDiscretizationKwargs'])
    generator_config = GENERATOR_CONFIGS[gc['GeneratorConfigType']](td, parameter_set, **gc['GeneratorConfigKwargs'])
    generator = GENERATORS[gc['GeneratorType']](generator_config)

    hc = config["Hedge"]
    strategy = StrategyNet(StrategyNetConfig(**hc['StrategyKwargs'], output_activation=nn.ReLU()))
    derivative = DERIVATIVES[hc['DerivativeType']](**hc['DerivativeKwargs'])
    objective = OBJECTIVES[hc['Objective']](**hc['ObjectiveKwargs'])
    masks = Masks(**{k: np.array(v) for k, v in hc['Masks'].items()})

    return DeepHedge(strategy, derivative, generator, objective, masks=masks)
