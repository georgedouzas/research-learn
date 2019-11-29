"""
The :mod:`rlearn.utils.validation` includes utilities
for input validation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from itertools import product

from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_random_state, check_X_y
from sklearn.model_selection._search import ParameterGrid
from imblearn.pipeline import Pipeline


def check_random_states(random_state, repetitions):
    """Create random states for experiments."""
    random_state = check_random_state(random_state)
    return [
        random_state.randint(0, 2 ** 32 - 1, dtype='uint32') for _ in range(repetitions)
    ]


def check_param_grids(param_grids, est_names):
    """Check the parameters grids to use with
    parametrized estimators."""

    # Check the parameters grids
    flat_param_grids = [
        param_grid for param_grid in list(ParameterGrid(param_grids)) if param_grid
    ]

    # Append existing estimators names
    param_grids = []
    for param_grid in flat_param_grids:

        # Get estimator name
        est_name = param_grid.pop('est_name', None)

        # Modify values
        param_grid = {param: [val] for param, val in param_grid.items()}

        # Check estimators prefixes
        params_prefixes = set([param.split('__')[0] for param in param_grid.keys()])
        if not params_prefixes.issubset(est_names):
            raise ValueError(
                'Parameters prefixes are not subset of parameter `est_names`.'
            )
        if len(params_prefixes) > 1:
            raise ValueError('Parameters prefixes are not unique.')
        if est_name is not None and len(params_prefixes.union([est_name])) > 1:
            raise ValueError(
                'Parameters prefixes and parameter `est_name` are not unique.'
            )
        param_grid['est_name'] = (
            [est_name] if est_name is not None else list(params_prefixes)
        )

        # Append parameter grid
        param_grids.append(param_grid)

    # Append missing estimators names
    current_est_names = set([param_grid['est_name'][0] for param_grid in param_grids])
    missing_est_names = set(est_names).difference(current_est_names)
    for est_name in missing_est_names:
        param_grids.append({'est_name': [est_name]})

    return param_grids


def check_oversamplers_classifiers(oversamplers, classifiers, random_state, n_runs):
    """Extract estimators and parameters grids."""

    # Create random states
    random_states = check_random_states(random_state, n_runs)

    # Create estimators and parameter grids
    estimators, param_grids = [], []
    for oversampler, classifier in product(oversamplers, classifiers):

        # Unpack oversamplers and classifiers
        ovs_name, ovs, ovs_param_grid = oversampler
        clf_name, clf, clf_param_grid = classifier
        if ovs is None:
            ovs = FunctionTransformer(validate=True)

        # Create estimator
        name = f'{ovs_name}|{clf_name}'
        ovs_steps = ovs.steps if isinstance(ovs, Pipeline) else [(ovs_name, ovs)]
        clf_steps = clf.steps if isinstance(clf, Pipeline) else [(clf_name, clf)]
        steps = ovs_steps + clf_steps
        estimators.append((name, Pipeline(steps)))

        # Create parameter grid
        ovs_prefix = f'{name}' if isinstance(ovs, Pipeline) else f'{name}__{ovs_name}'
        ovs_param_grid = [
            {f'{ovs_prefix}__{param}': val for param, val in param_grid.items()}
            for param_grid in ParameterGrid(ovs_param_grid)
        ]
        clf_prefix = f'{name}' if isinstance(clf, Pipeline) else f'{name}__{clf_name}'
        clf_param_grid = [
            {f'{clf_prefix}__{param}': val for param, val in param_grid.items()}
            for param_grid in ParameterGrid(clf_param_grid)
        ]
        combinations = product(ovs_param_grid, clf_param_grid, random_states)
        for param_grid1, param_grid2, random_state in combinations:
            param_grid1.update(param_grid2)
            param_grid = {'est_name': [name]}
            for param in ovs.get_params().keys():
                if 'random_state' in param:
                    param_grid.update({f'{ovs_prefix}__{param}': [random_state]})
            for param in clf.get_params().keys():
                if 'random_state' in param:
                    param_grid.update({f'{clf_prefix}__{param}': [random_state]})
            param_grid.update({param: [val] for param, val in param_grid1.items()})
            param_grids.append(param_grid)

    return estimators, param_grids


def check_datasets(datasets):
    """Check that datasets is a list of (dataset_name, (X, y)) pairs."""
    try:
        # Get datasets names
        datasets_names = [dataset_name for dataset_name, _ in datasets]

        # Check if datasets names are unique strings
        are_all_strings = all(
            [isinstance(dataset_name, str) for dataset_name in datasets_names]
        )
        are_unique = len(list(datasets_names)) == len(set(datasets_names))
        if not are_all_strings or not are_unique:
            raise ValueError('The datasets names should be unique strings.')
    except TypeError:
        raise TypeError(
            'The datasets should be a list ' 'of (dataset name, (X, y)) pairs.'
        )

    datasets = [(name, check_X_y(X, y)) for name, (X, y) in datasets]

    return datasets


def check_estimator_type(estimators):
    """Returns the type of estimators."""
    estimator_types = set([estimator._estimator_type for _, estimator in estimators])
    if len(estimator_types) > 1:
        raise ValueError(
            'Both classifiers and regressors were found. '
            'A single estimator type should be included.'
        )
    return estimator_types.pop()
