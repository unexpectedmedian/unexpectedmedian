import os
import math
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D
import numpy.polynomial.chebyshev as cheb


from pitutils import utils
from utils import ridge_regression_to_json, ridge_regression_from_json


def interpolate_forwards(fv):
    """
    let Nf =len(fv), dt = 1/Nf. then the interpolated function f would have

    f(x) = fv[i] for i/Nf <= x < (i+1)/Nf, i=0,...,Nf-1

    """
    Nf = len(fv)
    fv_ts = np.linspace(0, 1, Nf+1)[:-1]
    return interpolate.interp1d(fv_ts, fv, kind='previous', bounds_error=False,
                                fill_value=(fv[0], fv[-1]), assume_sorted=True)


def calculate_slope(fv, nt=101, fv_eps=0.1, fv_eps_scale=0.5, sigma=1.0, nsim=10000, seed=234234):
    """ returns d(median(fv*eps))/d(eps^2) (ie sensitivity to eps^2) around fv_eps
        Ie the slope of median(eps) vs eps^2 using fv_eps and fv_eps * fv_eps_scale
        For this to make sense we should remove a linear trend from fvs and scale
        them to unit-ish length
    """
    ts = np.linspace(0, 1, nt)
    dt = ts[1] - ts[0]
    fvs = interpolate_forwards(fv)(ts)
    np.random.seed(seed)

    zs = math.sqrt(dt)*np.random.normal(0.0, 1.0, (nsim//2, nt-1))
    paths = np.zeros((nsim, nt))
    paths[:nsim//2, 1:] = sigma*np.cumsum(zs, axis=1)
    # antithetic variables
    paths[nsim//2:, :] = -paths[:nsim//2, :]
    eps1 = fv_eps
    eps2 = eps1*fv_eps_scale
    # this broadcasts it fine
    paths1 = paths + eps1*fvs
    paths2 = paths + eps2*fvs

    med1 = np.mean(np.median(paths1, axis=1))/eps1
    med2 = np.mean(np.median(paths2, axis=1))/eps2

    # print('.', end='')

    return (med1 - med2)/(eps1**2 - eps2**2), (med1, med2, eps1, eps2)


def normalize_forwards(fv, do_normalize=False):

    if not do_normalize:
        return fv, (0.0, 0.0, 1.0)

    idx = np.arange(len(fv))
    slope, intercept, *_ = stats.linregress(idx, fv)
    fv -= (intercept + slope*idx)
    fv_scale = math.sqrt(sum(fv*fv))
    fv /= fv_scale
    return fv, (slope, intercept, fv_scale)


def cheb_fwd_xs(nf: int):
    """
    The grid we use to convert chebyshev coefficients to forward values and back
    Make it symmetric
    """
    return 2*np.arange(nf)/nf - 1 + 1/nf


def chebstate_to_forwards(state, do_normalize=False):
    """
    we return fwd values that correspond to scaled state (= coefs starting from the 2nd one, first two assume to be  zero)
    Let n=len(state). We set coefs = [0,0,state/norm(state)] and fv = chebval(range(n+2), coefs)

    If we want len(fv) = nf then we need len(state) = nf-2

    details
    https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebfit.html
    """
    nf = len(state)+2
    xs = cheb_fwd_xs(nf)
    state_norm = np.linalg.norm(state)
    if do_normalize:
        state = state/state_norm
    return cheb.chebval(xs, c=[0, 0, *state]), state_norm


def chebfit2(x, y, deg, alpha=1e-4):
    """
    This is a reimplementaion of a subset of chebyshev.chebfit
    https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebfit.html
    using Tichnov regularization (Ridge regression)
    """
    cheb_values = []
    for pos in range(deg+1):
        coefs = np.zeros(deg+1)
        coefs[pos] = 1
        vals = cheb.chebval(x, coefs)
        cheb_values.append(vals)
    cheb_values = np.array(cheb_values)
    regr = Ridge(alpha=alpha, fit_intercept=False)
    _ = regr.fit(cheb_values.T, y)
    return regr.coef_


def forwards_to_chebstate(fv, do_normalize=False, cdeg=None, rcond=1e-4):
    """
    We should get as many coefs as there are elements in fv.
    If a = chebstate[0] and b = chebstate[1] then fv[i] = a * b*i + ...
    where ... has mean zero and no slope so
    median(fv) ~ a + b/2 + median(...)
    """
    nf = len(fv)
    xs = cheb_fwd_xs(nf)

#    cb_coefs = cheb.chebfit(
#        xs, fv, deg=cdeg if cdeg is not None else len(xs)-1, rcond=rcond)
    cb_coefs = chebfit2(xs, fv, len(xs)-1, alpha=rcond)
    cb_redu = cb_coefs[2:]
    cb_norm = np.linalg.norm(cb_redu) if do_normalize else 1.0

    # we return coefs that can go into the trained model (first arg),
    # the scaling to be applied to the result of the model, and then lebel and slope
    # recall that median ( a + b*i + scale*(...) ) = a + b/2 + scale * median(...)
    return cb_redu/cb_norm, cb_norm, cb_coefs[0], cb_coefs[1]


def generate_cgrid_pairs_permutation(domain_params, n_fwds):
    # cmin, cmax, nc = domain_params
    # cgrid = np.linspace(cmin, cmax, nc)
    cgrid = np.linspace(*domain_params)
    cgrid2 = np.array([[c1, c2] for c1 in cgrid for c2 in cgrid])

    n_state_vars = n_fwds - 2
    all_pairs = list(itertools.combinations(range(n_state_vars), 2))

    block_len = cgrid2.shape[0]
    cgridall = np.zeros((len(all_pairs)*block_len, n_state_vars))
    for i, idx_pair in enumerate(all_pairs):
        cgridall[block_len*i:block_len*(i+1), idx_pair[0]] = cgrid2[:, 0]
        cgridall[block_len*i:block_len*(i+1), idx_pair[1]] = cgrid2[:, 1]

    all_pairs_exploded = [[pair]*block_len for pair in all_pairs]
    all_pairs_exploded = [
        item for sublist in all_pairs_exploded for item in sublist]
    return cgridall, all_pairs_exploded


def cheb_coef_name_for_index(idx):
    return f'chb{idx+2:02d}'


def forwards_and_values_df_from_cheb_coefs(
        cheb_coefs, do_normalize=False, slope_calc_args: dict = None, save_file=None,
        extra_cols: dict = None):
    n_state_vars = cheb_coefs.shape[1]
    n_fwds = n_state_vars + 2

    # normalize
    cheb_coefs_norm = np.linalg.norm(cheb_coefs, axis=1)

    if do_normalize:
        for i in range(cheb_coefs.shape[0]):
            cheb_coefs[i, :] /= cheb_coefs_norm[i]

    train_F = [chebstate_to_forwards(coefs, do_normalize=do_normalize)[
        0] for coefs in cheb_coefs]
    vals = [calculate_slope(fv, **slope_calc_args) for fv in tqdm(train_F)]

    df = pd.DataFrame(
        columns=[cheb_coef_name_for_index(n) for n in range(n_state_vars)], data=cheb_coefs)
    df['value'] = [v[0] for v in vals]

    df[[f'fwd{n:02d}' for n in range(n_fwds)]] = train_F
    df['scale_cheb'] = cheb_coefs_norm
    df['med0'] = [v[1][0] for v in vals]

    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v

    if save_file:
        df.to_csv(save_file)

    return df


def value_from_linear_cubed_model(
        cheb_coefs, do_normalize=False, linear_model_file_name='./data_ml/lin_regr_cubed_fwds10_01.json', save_file=None):

    with open(linear_model_file_name, 'r') as f:
        model_json = f.read()
    linear_cubed_model = ridge_regression_from_json(model_json)

    n_rows = cheb_coefs.shape[0]
    n_state_vars = cheb_coefs.shape[1]
    n_fwds = n_state_vars + 2

    # normalize
    cheb_coefs_norm = np.linalg.norm(cheb_coefs, axis=1)

    if do_normalize:
        for i in range(n_rows):
            cheb_coefs[i, :] /= cheb_coefs_norm[i]

    cheb_cols = [cheb_coef_name_for_index(n) for n in range(n_state_vars)]
    df = pd.DataFrame(columns=cheb_cols, data=cheb_coefs)

    fwds_cols = [f'fwd{n:02d}' for n in range(n_fwds)]
    perms_of_3s = list(itertools.combinations_with_replacement(fwds_cols, 3))

    if len(perms_of_3s) != len(linear_cubed_model.coef_):
        raise Exception((f'Forward dimension mismatch, passed in '
                         f'{n_fwds} forwards and obtained {len(fwds_cols)} cubed columns but'
                         f'n_coefs = {len(linear_cubed_model.coef_)}'))

    fwds = [chebstate_to_forwards(coefs, do_normalize=do_normalize)[
        0] for coefs in cheb_coefs]
    df[fwds_cols] = fwds

    # fwds_cubed = {':'.join(p): df[[*p]].prod(axis=1) for p in perms_of_3s}
    # cubed_X = pd.DataFrame(fwds_cubed)

    fwds_cubed = np.concatenate(
        [df[[*p]].prod(axis=1).values.reshape(n_rows, 1) for p in perms_of_3s], axis=1)

    vals = linear_cubed_model.predict(fwds_cubed)
    df['value'] = vals
    df['scale_cheb'] = cheb_coefs_norm
    # do not have this
    df['med0'] = [np.nan]*n_rows

    if save_file:
        df.to_csv(save_file)

    return df


def generate_training_set_cheb(
        n_fwds, n_samples, do_normalize=False, sample_seed=997986, slope_calc_args: dict = None, save_file=None):

    if not slope_calc_args:
        slope_calc_args = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                               sigma=1.0, nsim=1000, seed=234234)

    n_state_vars = n_fwds - 2
    zs = np.random.uniform(
        low=-1.0, high=1.0, size=(n_samples//2, n_state_vars))
    # antithetic
    zs = np.concatenate((zs, -zs), axis=0)

    return forwards_and_values_df_from_cheb_coefs(zs, do_normalize=do_normalize, slope_calc_args=slope_calc_args, save_file=save_file)


def generate_testing_set_cheb(
        n_fwds, domain_params=(-0.75, 0.75, 11), do_normalize=False, slope_calc_args: dict = None, save_file=None):
    if not slope_calc_args:
        slope_calc_args = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                               sigma=1.0, nsim=1000, seed=234234)

    cgridall, all_pairs = generate_cgrid_pairs_permutation(
        domain_params, n_fwds)
    test_df = forwards_and_values_df_from_cheb_coefs(
        cgridall, do_normalize=do_normalize, slope_calc_args=slope_calc_args,
        save_file=save_file, extra_cols={'cheb_indices': all_pairs}
    )

    return test_df


def generate_training_set_simple(
        n_fwds, n_samples, do_normalize=False, return_extra=True, sample_seed=997986, slope_calc_args: dict = None):

    if not slope_calc_args:
        slope_calc_args = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                               sigma=1.0, nsim=1000, seed=234234)

    zs = np.random.normal(0.0, 1.0, (n_samples//2, n_fwds))

    # antithetic
    zs = np.concatenate((zs, -zs), axis=0)

    zs_normalized = [normalize_forwards(
        zs[n, :], do_normalize) for n in range(zs.shape[0])]
    train_X = [zsn[0] for zsn in zs_normalized]
    train_X_scale_params = [zsn[1] for zsn in zs_normalized]
    vals = [calculate_slope(fv, **slope_calc_args) for fv in train_X]

    train_df = pd.DataFrame(
        columns=[f'fwd{n:02d}' for n in range(n_fwds)], data=train_X)

    train_df['value'] = [v[0] for v in vals]

    if return_extra:
        train_df['slp'] = [sp[0] for sp in train_X_scale_params]
        train_df['itr'] = [sp[1] for sp in train_X_scale_params]
        train_df['scl'] = [sp[2] for sp in train_X_scale_params]
        train_df['med0'] = [v[1][0] for v in vals]

    return train_df


def filename_for_params(
        prefix: str, n_fwds: int, n_samples: int, do_normalize: bool, slope_params: dict, postfix: str = ''):
    nt = slope_params['nt']
    nsim = slope_params['nsim']
    return f'{prefix}_nf{n_fwds}_nsmpl{n_samples}_norm{int(do_normalize)}_nt{nt}_nsim{nsim}{postfix}.csv'


def do_generate_training_set():
    n_fwds, n_samples, do_normalize, sample_seed = 20, 25000, False, 123123

    slope_params = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                        sigma=1.0, nsim=100000, seed=234234)

    this_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(this_folder, 'data_ml')
    filename = filename_for_params(
        'training', n_fwds, n_samples, do_normalize, slope_params)
    filepath = utils.get_nonexistent_path(os.path.join(data_folder, filename))

    train_df = generate_training_set_cheb(
        n_fwds, n_samples, do_normalize=do_normalize, sample_seed=sample_seed, slope_calc_args=slope_params, save_file=filepath)

    print('done')

    print('description stats for Y')
    print(train_df['value'].describe())
    return train_df


def do_generate_training_set_multi(n_processes: int):
    import multiprocessing
    slope_params = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                        sigma=1.0, nsim=1000000, seed=234234)

    np.random.seed(927234)
    seeds = [np.random.randint(100000, 200000) for _ in range(n_processes)]

    ps = []
    for n in range(n_processes):
        n_fwds, n_samples, do_normalize, sample_seed = 20, 5000, False, seeds[n]
        this_folder = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(this_folder, 'data_ml')

        filename = filename_for_params(
            'training', n_fwds, n_samples, do_normalize, slope_params, f'_p{n}')
        filepath = utils.get_nonexistent_path(
            os.path.join(data_folder, filename))

        ps.append(multiprocessing.Process(
            target=generate_training_set_cheb,
            args=(n_fwds, n_samples, do_normalize,
                  sample_seed, slope_params, filepath)
        ))

    for n in range(n_processes):
        ps[n].start()

    for n in range(n_processes):
        ps[n].join()
        print(f'Process {n} is done')

    print('All done')


def do_generate_testing_set():
    n_fwds = 10
    domain_params = (-1.0, 1.0, 21)
    do_normalize = False

    # n_samples is for filename only
    n_cheb_coefs = n_fwds-2
    n_pairs = n_cheb_coefs * (n_cheb_coefs - 1)/2
    n_samples = n_pairs * domain_params[2]*domain_params[2]

    slope_params = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                        sigma=1.0, nsim=1000000, seed=234234)

    this_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(this_folder, 'data_ml')
    filename = filename_for_params(
        'testing', n_fwds, n_samples, do_normalize, slope_params)
    filepath = utils.get_nonexistent_path(os.path.join(data_folder, filename))

    test_df = generate_testing_set_cheb(
        n_fwds, domain_params=domain_params, do_normalize=do_normalize, slope_calc_args=slope_params, save_file=filepath)

    print('done')

    print('description stats for Y')
    print(test_df['value'].describe())
    return test_df


def do_generate_testing_set_multi(n_processes_per_run: int, n_runs: int = 1):
    import multiprocessing

    n_processes = n_processes_per_run * n_runs

    n_fwds = 20
    domain_params = (-1.0, 1.0, 21)
#    domain_params = (-1.0, 1.0, 3)  # for testing the testing
    do_normalize = False

    # n_samples is for filename only
    n_cheb_coefs = n_fwds-2
    n_pairs = n_cheb_coefs * (n_cheb_coefs - 1)//2
    n_samples = n_pairs * domain_params[2]*domain_params[2]

    slope_calc_args = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                           sigma=1.0, nsim=1000000, seed=234234)

    cgridall, all_pairs = generate_cgrid_pairs_permutation(
        domain_params, n_fwds)

    # just to make sure
    if n_samples != len(all_pairs):
        raise ValueError('length mismatch')

    ns_per_batch = int(math.ceil(n_samples/n_processes))
    cgrid_batches = [
        cgridall[n*ns_per_batch:min((n+1)*ns_per_batch, n_samples), :] for n in range(n_processes)]
    all_pairs_batches = [
        all_pairs[n*ns_per_batch:min((n+1)*ns_per_batch, n_samples)] for n in range(n_processes)]

    for r in range(n_runs):
        ps = []
        for p in range(n_processes_per_run):
            n = r*n_processes_per_run + p
            act_n_samples = len(all_pairs_batches[n])

            this_folder = os.path.dirname(os.path.abspath(__file__))
            data_folder = os.path.join(this_folder, 'data_ml')

            filename = filename_for_params(
                'testing', n_fwds, act_n_samples, do_normalize, slope_calc_args, f'_p{n}')
            filepath = utils.get_nonexistent_path(
                os.path.join(data_folder, filename))

            ps.append(multiprocessing.Process(
                target=forwards_and_values_df_from_cheb_coefs,
                args=(cgrid_batches[n], do_normalize, slope_calc_args, filepath,
                      {'cheb_indices': all_pairs_batches[n]}
                      )
            ))

        for p in range(n_processes_per_run):
            ps[p].start()

        for p in range(n_processes_per_run):
            ps[p].join()
            print(f'Run {r} Process {p} is done')

    print('All done')


def test_interp_01():
    fv = [1, 2, 3, 4, 5]
    ts = np.linspace(-1, 2, 31)

    fv_all = interpolate_forwards(fv)(ts)
    plt.plot(ts, fv_all, '.')
    plt.show()


def test_slope_01(nsim=10000, seed=234235):
    fv = [0, 0, 0, 0, 0]
    slope, others = calculate_slope(fv, nsim=nsim, seed=seed)
    print(slope, others)

    fv = np.arange(101)/10
    slope, others = calculate_slope(fv, nsim=nsim, seed=seed)
    print(slope, others)

    fv = [1, 1, 3, 1, 1]
    slope, others = calculate_slope(fv, nsim=nsim, seed=seed)
    print(slope, others)

    fv = [0, 0, 2, 0, 0]
    slope, others = calculate_slope(fv, nsim=nsim, seed=seed)
    print(slope, others)


def test_gen_training_set_simple_01():
    n_fwds, n_samples, do_normalize, sample_seed = 3, 100, False, 123123

    slope_params = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                        sigma=1.0, nsim=100000, seed=234234)
    train_df = generate_training_set_simple(
        n_fwds, n_samples, do_normalize=do_normalize, sample_seed=sample_seed, slope_calc_args=slope_params)

    filename = './data_ml/traing_res_test_01.csv'
    train_df.to_csv(filename)

    fv1 = train_df['fwd01'] - train_df['fwd00']
    fv2 = train_df['fwd02'] - train_df['fwd00']
    val = train_df['value'] - train_df['fwd00']

    slp = fv2
    crv = 2 * fv1 - fv2

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(slp.values, crv.values, val.values, marker='.')

    plt.show()
    print(slp)


def test_gen_training_set_cheb_01(save_file='./data_ml/traing_res_test_03.csv'):
    n_fwds, n_samples, do_normalize, sample_seed = 4, 1000, False, 123123

    slope_params = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                        sigma=1.0, nsim=1000, seed=234234)
    train_df = generate_training_set_cheb(
        n_fwds, n_samples, do_normalize=do_normalize, sample_seed=sample_seed, slope_calc_args=slope_params)

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_df['chb02'], train_df['chb03'],
               train_df['value'], marker='.')

    plt.show()


def test_gen_testing_set_cheb_01(save_file='./data_ml/traing_res_test_04.csv'):
    n_fwds, domain_params, do_normalize = 4, (-1, 1, 21), False

    slope_params = dict(nt=101, fv_eps=0.1, fv_eps_scale=0.5,
                        sigma=1.0, nsim=1000, seed=234234)
    test_df = generate_testing_set_cheb(
        n_fwds, domain_params, do_normalize=do_normalize, slope_calc_args=slope_params, save_file=save_file)

    idx = (0, 1)
    idx_names = [cheb_coef_name_for_index(i) for i in idx]
    sub_df = test_df[test_df['cheb_indices'] == idx]

    ncsq = len(sub_df)
    nc = int(math.sqrt(ncsq))
    coefs = sub_df[idx_names].values
    coefs1 = coefs[:, 0].reshape((nc, nc))
    coefs2 = coefs[:, 1].reshape((nc, nc))
    vals2d = sub_df['value'].values.reshape((nc, nc))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
#    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(coefs1, coefs2, vals2d)
    plt.show()


def test_fit_01(load_file='./data_ml/traing_res_test_02.csv'):
    from tensorflow import keras

    train_df = pd.read_csv(load_file, index_col=0)

    n_rows = len(train_df)
    n_epochs = 10
    batch_size = n_rows  # None #nSim//128
    learn_rate = 0.01  # 0.001

    input_cols = train_df.columns[train_df.columns.str.startswith('chb')]
    inputX = train_df[input_cols]
    inputY = train_df['value']
    n_state_vars = len(inputX.columns)

    nodes0 = 5
    nodes1 = 5
    model = keras.Sequential()

    model = keras.Sequential()
    model.add(keras.layers.Dense(
        nodes0, input_dim=n_state_vars, activation='relu'))
    model.add(keras.layers.Dense(nodes1, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    opt = keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mae'])

    stats_before = model.evaluate(inputX, inputY)
    model.fit(inputX, inputY, epochs=50*n_epochs, batch_size=batch_size)
    stats_after = model.evaluate(inputX, inputY)

    print(stats_before)
    print(stats_after)

    fit = model.predict(inputX)

    plt.plot(inputY, fit, '.')
    plt.show()

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_df['chb02'], train_df['chb03'],
               inputY, marker='.')
    ax.scatter(train_df['chb02'], train_df['chb03'],
               fit, marker='.')

    plt.show()


def test_generate_cgrid_pairs_permutation():
    res, pairs = generate_cgrid_pairs_permutation((0.5, 1, 3), 8)
    print(res)
    print(pairs)


def test_linear_cubed_model_01():
    res, idcs = generate_cgrid_pairs_permutation((-1, 1, 21), n_fwds=10)
    test_df = value_from_linear_cubed_model(
        res, do_normalize=False, linear_model_file_name='./data_ml/lin_regr_cubed_fwds10_01.json', save_file=None)
    test_df['cheb_indices'] = idcs

    idx = (0, 1)
    idx_names = [cheb_coef_name_for_index(i) for i in idx]
    sub_df = test_df[test_df['cheb_indices'] == idx]

    ncsq = len(sub_df)
    nc = int(math.sqrt(ncsq))
    coefs = sub_df[idx_names].values
    coefs1 = coefs[:, 0].reshape((nc, nc))
    coefs2 = coefs[:, 1].reshape((nc, nc))
    vals2d = sub_df['value'].values.reshape((nc, nc))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(coefs1, coefs2, vals2d)
    plt.show()


if __name__ == '__main__':
    # test_interp_01()
    # test_slope_01(nsim=1000000, seed=234235)
    # test_gen_training_set_simple_01()
    # test_gen_training_set_cheb_01()
    # test_gen_testing_set_cheb_01()
    # test_fit_01()
    # test_generate_cgrid_pairs_permutation()
    # test_linear_cubed_model_01()
    # do_generate_training_set()
    # do_generate_training_set_multi(n_processes=5)
    do_generate_testing_set_multi(n_processes_per_run=5, n_runs=5)

    # do_generate_testing_set()
