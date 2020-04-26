import sys
from pywebcachesim import parser, database, runner, get_task
import yaml
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.core import ParameterSpace, ContinuousParameter
from copy import deepcopy
import numpy as np
from sklearn.linear_model import LinearRegression


def update_by_file(args, filename):
    args_cp = deepcopy(args)
    # job config file
    with open(filename) as f:
        file_params = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in file_params.items():
        if args_cp.get(k) is None:
            args_cp[k] = v
    return args_cp


def _get_task(trace_file, cache_type, cache_size, parameters, n_early_stop, memory_window):
    task = {
        **parameters,
        'trace_file': trace_file,
        'cache_type': cache_type,
        'cache_size': cache_size,
        'n_early_stop': n_early_stop,
        'memory_window': memory_window,
    }
    return task


def get_validation_tasks_per_cache_size(trace_file, cache_type, cache_size, parameters, args_global, df):
    n_req = parameters['n_req']
    n_validation = int(n_req * args_global['ratio_validation'])
    n_iteration = args_global['n_iteration']
    n_beam = args_global['n_beam']
    if len(df) == 0 or len(df[df['cache_size'] == cache_size]) == 0:
        # init value
        next_windows = np.linspace(1, int(0.4 * n_validation), args_global['n_beam'] + 1, dtype=int)[1:]
        tasks = []
        for memory_window in next_windows:
            # override n_early stop
            task = _get_task(trace_file, cache_type, cache_size, parameters, n_validation, memory_window)
            tasks.append(task)
        return tasks

    # as emukit output is random, don't assume same points each time, as long as # point is enough
    df1 = df[df['cache_size'] == cache_size]
    if len(df1) >= n_iteration * n_beam:
        return []
    # next round
    # window at most 40% of length
    # add xs, ys in a consistent order otherwise the results will be difference
    parameter_space = ParameterSpace([ContinuousParameter('x', 1, int(0.4 * n_validation))])
    bo = GPBayesianOptimization(variables_list=parameter_space.parameters,
                                X=df1['memory_window'].values.reshape(-1, 1),
                                Y=df1['byte_miss_ratio'].values.reshape(-1, 1),
                                batch_size=args_global['n_beam'])
    next_windows = bo.suggest_new_locations().reshape(-1).astype(int)
    tasks = []
    for memory_window in next_windows:
        task = _get_task(trace_file, cache_type, cache_size, parameters, n_validation, memory_window)
        tasks.append(task)
    return tasks


def check_need_fitting(cache_size, parameters, args_global, df):
    df1 = df[df['cache_size'] == cache_size]
    candidate = df1['memory_window'].iloc[0]
    # You could try different memory window setting to see if you can find a U-curve: y-axis is the byte miss ratio
    # and X-axis is the memory window. Only when the algorithm is confident about such U curve happens, it will select
    # that candidate The "if candidate .." line is to select that. If there is no such U-curve, no memory window will
    # be selected at current cache size
    if candidate == df1['memory_window'].max():
        # we don't accept the memory window at right boundary, as it may not be a local minimum
        return True
    else:
        return False


def get_evaluation_task(trace_file, cache_type, cache_size, parameters, args_global, df):
    memory_window = df[df['cache_size'] == cache_size]['memory_window'].iloc[0]
    return _get_evaluation_task(trace_file, cache_type, cache_size, memory_window, parameters)


def _get_evaluation_task(trace_file, cache_type, cache_size, memory_window, parameters):
    n_early_stop = parameters['n_early_stop']
    # TODO: ideally need to match all parameters. So this means doesn't support multiple values for same lrb
    #  parameter
    df = database.load_reports(
        trace_file=trace_file,
        cache_type=cache_type,
        cache_size=str(cache_size),
        n_early_stop=str(n_early_stop),
        n_warmup=0,
        memory_window=str(memory_window),
        version=parameters['version'],  # use version as a strong key
        dburi=parameters["dburi"],
        dbcollection=parameters["dbcollection"],
    )
    if len(df) != 0:
        # test this before
        return []
    else:
        task = _get_task(trace_file, cache_type, cache_size, parameters, n_early_stop, memory_window)
        return [task]


def get_fitting_task(trace_file, cache_type, cache_size, parameters, args_global, df):
    n_fitting_points = args_global['n_fitting_points']
    current_cache_size = cache_size // 2
    xs = []
    ys = []
    while len(xs) != n_fitting_points:
        tasks = get_validation_tasks_per_cache_size(trace_file, cache_type, current_cache_size, parameters, args_global,
                                                    df)
        if len(tasks) == 0:
            need_fitting = check_need_fitting(current_cache_size, parameters, args_global, df)
            if need_fitting == False:
                window = df[df['cache_size'] == current_cache_size]['memory_window'].iloc[0]
                xs.append(current_cache_size)
                ys.append(window)
            # recursively add a smaller cache size
            current_cache_size = current_cache_size // 2
        else:
            return tasks
    reg = LinearRegression(fit_intercept=False).fit(np.array(xs).reshape(-1, 1), np.array(ys))
    memory_window = int(reg.predict([[cache_size]])[0])
    return _get_evaluation_task(trace_file, cache_type, cache_size, memory_window, parameters)


def get_tasks_per_cache_size(trace_file, cache_type, cache_size, parameters, args_global):
    n_req = parameters['n_req']
    n_validation = int(n_req * args_global['ratio_validation'])
    n_warmup = int(0.8 * n_validation)
    # TODO: ideally need to match all parameters. So this means doesn't support multiple values for same lrb
    #  parameter
    df = database.load_reports(
        trace_file=trace_file,
        cache_type=cache_type,
        # no cache size because may check smaller cache sizes
        n_early_stop=str(n_validation),
        n_warmup=n_warmup,
        version=parameters['version'],  # use version as a strong key
        dburi=parameters["dburi"],
        dbcollection=parameters["dbcollection"],
    )
    tasks = get_validation_tasks_per_cache_size(trace_file, cache_type, cache_size, parameters, args_global, df)
    if len(tasks) != 0:
        return tasks
    need_fitting = check_need_fitting(cache_size, parameters, args_global, df)
    if need_fitting == False:
        return get_evaluation_task(trace_file, cache_type, cache_size, parameters, args_global, df)
    return get_fitting_task(trace_file, cache_type, cache_size, parameters, args_global, df)


def get_cache_size_and_parameter_list(trace_file, cache_type, cache_size_or_size_parameters, args_size, args_global):
    # element can be k: v or k: list[v], which would be expanded with cartesian product
    # priority: global < default < per trace < per trace per algorithm < per trace per algorithm per cache size
    parameters = {}
    # global
    for k, v in args_global.items():
        if k not in [
            'cache_types',
            'trace_files',
            'algorithm_param_file',
            'trace_param_file',
            'job_file',
            'debug',
            'nodes',
            'n_iteration',
            'n_beam',
            'n_fitting_points',
            'ratio_validation',
        ] and v is not None:
            parameters[k] = v
    # default
    default_algorithm_params = update_by_file({}, args_size['algorithm_param_file'])
    if cache_type in default_algorithm_params:
        parameters = {**parameters, **default_algorithm_params[cache_type]}
    # per trace
    for k, v in args_size[trace_file].items():
        if k not in ['cache_sizes'] and k not in default_algorithm_params and v is not None:
            parameters[k] = v
    # per trace per algorithm
    if cache_type in args_size[trace_file]:
        # trace parameters overwrite default parameters
        parameters = {**parameters, **args_size[trace_file][cache_type]}
    # per trace per algorithm per cache size
    if isinstance(cache_size_or_size_parameters, dict):
        # only 1 key (single cache size) is allowed
        assert (len(cache_size_or_size_parameters) == 1)
        cache_size = list(cache_size_or_size_parameters.keys())[0]
        if cache_type in cache_size_or_size_parameters[cache_size]:
            # per cache size parameters overwrite other parameters
            parameters = {**parameters, **cache_size_or_size_parameters[cache_size][cache_type]}
    else:
        cache_size = cache_size_or_size_parameters
    parameter_list = get_task.cartesian_product(parameters)
    return cache_size, parameter_list


def get_tasks(args):
    """
     convert job config to list of task
     @:returns dict/[dict]
     """
    # current version is only for LRB
    assert args['cache_types'] == ['LRB']

    tasks = []
    for trace_file in args['trace_files']:
        for cache_type in args['cache_types']:
            args_csize = update_by_file(args, args['trace_param_file'])
            for cache_size_or_size_parameters in args_csize[trace_file]['cache_sizes']:
                cache_size, parameter_list = get_cache_size_and_parameter_list(trace_file, cache_type,
                                                                               cache_size_or_size_parameters,
                                                                               args_csize, args)
                for parameters in parameter_list:
                    assert 'memory_window' not in parameters
                    _tasks = get_tasks_per_cache_size(trace_file, cache_type, cache_size, parameters, args)
                    # for tuning task, ignore n_early_stop, for eval task, use n_early_stop
                    tasks.extend(_tasks)
    # deduplicate tasks
    tasks = [dict(t) for t in {tuple(d.items()) for d in tasks}]
    return tasks


def main():
    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
        raise Exception('Error: python version need to be at least 3.6')
    args = parser.parse_cmd_args()
    args = update_by_file(args, args['job_file'])

    while True:
        tasks = get_tasks(args)
        if len(tasks) == 0:
            break
        if args["debug"]:
            print(tasks)
        runner.run(args, tasks)


if __name__ == '__main__':
    main()
