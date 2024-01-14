import random


def grow():
    """Grow strategy: always grow"""
    while True:
        yield 'grow'


def prune():
    """Prune strategy: always prune"""
    while True:
        yield 'prune'


def alternate():
    """Alternate strategy: grow, prune, grow, prune, ..."""
    while True:
        yield 'grow'
        yield 'prune'


def random_strategy():
    """Random strategy: grow or prune with equal probability"""
    while True:
        yield random.choice(['grow', 'prune'])


# Add your own strategy here! Don't forget to add it to the strategies dict below.

strategies = {
    'grow': grow,
    'prune': prune,
    'alternate': alternate,
    'random': random_strategy,
}
