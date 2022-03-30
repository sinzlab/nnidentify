import numpy as np

from insilico_stimuli.stimuli import *
from insilico_stimuli.parameters import *

def find_optimal_gabor_bruteforce(dataloaders, model, method_config):
    # Determine input size
    batch = next(iter(dataloaders['test'][method_config['data_key']]))
    _, _, w, h = batch[0].shape

    GaborSet(method_config['gabor_config'])

    # Finite Set
    canvas_size = [w, h]
    sizes = FiniteParameter([float(val) for val in range(
                            method_config['sizes']['low'],
                            method_config['sizes']['high'])]
                            [::method_config['sizes']['step']]
                            )

    method_config = {
        "sizes": FiniteParameter([float(val) for val in range(0, 10)][::1])
    }

    spatial_frequencies = FiniteParameter([float(val) for val in np.linspace(
                            method_config['spatial_frequencies']['low'],
                            method_config['spatial_frequencies']['high'],
                            method_config['spatial_frequencies']['step']
                            )])

    contrasts = FiniteParameter([method_config['contrasts']])

    orientations = FiniteParameter([float(val) for val in np.linspace(
        method_config['orientations']['low'],
        method_config['orientations']['high'],
        method_config['orientations']['step']
    )])

    phases = FiniteParameter([float(val) for val in np.linspace(
        method_config['phases']['low'],
        method_config['phases']['high'],
        method_config['phases']['step']
    )])

    grey_levels = FiniteParameter([method_config['grey_levels']])

    eccentricities = FiniteParameter([float(val) for val in np.linspace(
        method_config['eccentricities']['low'],
        method_config['eccentricities']['high'],
        method_config['eccentricities']['step']
    )])

    locations = FiniteParameter([[float(x), float(y)]
                                 for x in range(method_config['locations']['x']['low'], method_config['locations']['x']['high'])
                                 for y in range(method_config['locations']['y']['low'], method_config['locations']['y']['high'])]
                                [::method_config['locations']['step']])

    gabor_set = GaborSet(canvas_size=canvas_size,
                         locations=locations,
                         sizes=sizes,
                         spatial_frequencies=spatial_frequencies,
                         contrasts=contrasts,
                         orientations=orientations,
                         phases=phases,
                         grey_levels=grey_levels,
                         eccentricities=eccentricities)

    best_params, values = gabor_set.find_optimal_stimulus_bruteforce(
        model=model,
        data_key=method_config['data_key'],
        batch_size=method_config['batch_size']
    )

    return best_params, values