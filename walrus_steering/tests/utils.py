import itertools
from typing import Dict, List


def generate_parameters(conf_options: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Generate all the possible combination of options.
    It assumes the `conf_options` describes the possible values for each option.
    """
    conf_combinations = []
    for combination in itertools.product(*conf_options.values()):
        conf_combinations.append(dict(zip(conf_options.keys(), combination)))
    return conf_combinations
