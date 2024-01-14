import sys
import pathlib
if sys.version_info >= (3, 11):
    import tomllib as toml
else:
    import tomli as toml

import torch
from sad_nns.experiments.north_and_edl.north.grow_prune_strategy import strategies as grow_prune_strategies
from sad_nns.experiments.north_and_edl.north.grow_prune_metric import metrics as grow_prune_metrics
from sad_nns.experiments.data import Dataset
from sad_nns.uncertainty import MaximumLikelihoodLoss, CrossEntropyBayesRisk, SquaredErrorBayesRisk


def get_activations(model, layer):
    return model.activations[str(layer)]


def get_weights(model, layer):
    return model[layer].weight


class Config():
    def __init__(self, config_file):
        with open(config_file, 'rb') as config_file:
            self.config = toml.load(config_file)
        self.parse_config()

    def parse_config(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        hyperparameters = self.config['hyperparameters']
        self.learning_rate = hyperparameters['learning_rate']
        self.epochs = hyperparameters['epochs']
        self.batch_size = hyperparameters['batch_size']

        # Dataset
        data = self.config['data']
        self.dataset = Dataset(data['dataset'], data['image_size'], self.batch_size, 
                               split=data['train_val_split'])

        self.parse_edl()
        self.parse_north()

    def parse_edl(self):
        # One of 'max_likelihood_loss', 'cross_entropy_bayes_risk', or 'squared_error_bayes_risk'
        criterion = self.config['edl']['criterion']
        if criterion == 'max_likelihood_loss':
            self.criterion = MaximumLikelihoodLoss()
        elif criterion == 'cross_entropy_bayes_risk':
            self.criterion = CrossEntropyBayesRisk()
        elif criterion == 'squared_error_bayes_risk':
            self.criterion = SquaredErrorBayesRisk()
        else:
            raise ValueError(f'Invalid criterion {criterion}')

    def parse_north(self):
        north = self.config['north']
        grow_prune_strategy = north['grow_prune_strategy']
        self.grow_prune_strategy = grow_prune_strategies[grow_prune_strategy]()
        
        self.grow_prune_cycles = north['grow_prune_cycles']
        self.epochs_per_cycle = north['epochs_per_cycle']

        # Parse grow and prune metrics
        grow_metric_name = north['grow_metric']
        self.grow_metric = grow_prune_metrics.get(grow_metric_name, None)
        if self.grow_metric is None:
            raise ValueError(f'Invalid grow metric {grow_metric_name}')

        prune_metric_name = north['prune_metric']
        self.prune_metric = grow_prune_metrics.get(prune_metric_name, None)
        if self.prune_metric is None:
            raise ValueError(f'Invalid prune metric {prune_metric_name}')

        # replace any tensor configurations with `get_activations` or `get_weights`
        metric_params = north['metric_params']
        for metric, params in metric_params.items():
            if 'tensor' not in params:
                raise ValueError(f'No tensor specified for metric {metric}')

            if params['tensor'] == 'activation':
                params['tensor'] = get_activations
            elif params['tensor'] == 'weight':
                params['tensor'] = get_weights
            else:
                raise ValueError(f"Invalid tensor '{params['tensor']}' specified for metric '{metric}'")

        self.grow_metric_params = metric_params[grow_metric_name]
        self.prune_metric_params = metric_params[prune_metric_name]


config = Config(pathlib.Path(__file__).parent / 'config.toml')
