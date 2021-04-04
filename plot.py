import matplotlib.pyplot as plt
import numpy as np

def plot_cluster_stats(name, c_name, stats):
    fig, axes = plt.subplots(1, len(stats), figsize=(20,5))
    fig.suptitle(f'Num of clusters comparison for {name} and {c_name}')
    for i, score in enumerate(stats.items()):
        score_name, score_values = score
        ax = axes[i]
        ax.grid()
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel(score_name)
        x = list(range(2, len(score_values)+2))
        ax.plot(x, score_values, label=score_name, lw=2)
        ax.set_xticks(x)
        ax.legend(loc='best')
    save_fig(f'{name}/{c_name}')

def plot_cluster_stats_batch(name, c_name, stats):
    n = len(stats['PCA'])
    fig, axes = plt.subplots(2, n-2, figsize=(10,10))
    faxes = axes.flatten()
    fig.suptitle(f'Num of clusters comparison for {name} and {c_name}')
    for k in stats:
        for i, score in enumerate(stats[k].items()):
            score_name, score_values = score
            ax = faxes[i]
            ax.grid()
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel(score_name)
            x = list(range(2, len(score_values)+2))
            ax.plot(x, score_values, label=k, lw=2)
            ax.set_xticks(x)
            ax.legend(loc='best')
    save_fig(f'{name}_{c_name}')

def plot_rec_map(name, rec_map, special_rp):
    fig, ax = plt.subplots()
    fig.suptitle(f'Reconstruction error comparison for {name}')
    ax.grid()
    ax.set_xlabel('# of dimensions')
    ax.set_ylabel('Reconstruction error')
    for c_name in rec_map:
        rec = rec_map[c_name]
        x = list(range(1, len(rec) + 1))
        ax.plot(x, rec, label=c_name)
    if special_rp is not None:
        mean, std = special_rp
        rec = mean
        x = list(range(1, len(rec) + 1))
        ax.plot(x, rec, label='RP')
        ax.fill_between(x, mean - std, mean + std, alpha=0.4, color='purple')

    ax.set_xticks(x)
    ax.legend(loc='best')
    save_fig(f'{name}_rec')

def plot_cluster(name, c_name, X, y, k, tY, weights=None):
    from itertools import combinations
    if weights is not None:
        weights = np.asarray(weights)
        weights -= weights.min()
        weights /= weights.max()
        weights = .5 + weights / 2
    tY = tY.map({False: 'green', True: 'red'})
    f = list(combinations(range(0, X.shape[1]), 2))
    n = len(f)
    fx = fy = int(np.sqrt(n))
    if fx * fy < n:
        if fx > 2:
            fy += 1
        else:
            fx += 1
    fig, axes = plt.subplots(fx, fy, figsize=(10, 10) if fx < 8 else (20,20))
    fig.tight_layout(h_pad=2)
    faxes = axes.flatten()
    X = X[:300, :]
    y = y[:300]
    tY = tY[:300]
    if weights is not None:
        weights = weights[:300]
    for i, pair in enumerate(f):
        a, b = pair
        ax = faxes[i]
        ax.set_title(str(pair))
        ax.scatter(X[:, a], X[:, b], c=y, edgecolors=tY, alpha=1 if weights is None else weights)
#    fig.suptitle(f'Pair plots for {name} with {c_name} and k={k}')
#    plt.subplots_adjust(top=.2)
    save_fig(f'scatter/{name}_{c_name}/{k}')

def plot_simple(name, fname, title, x_label, y_label, values):
    fig, ax = plt.subplots()
    if title is not None:
        fig.suptitle(f'{title} for {name}')
    ax.grid()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    x = list(range(1, len(values)+1))
    ax.plot(x, values)
    ax.set_xticks(x)
    save_fig(f'{name}_{fname}')

def save_fig(name):
    import os
    path = f'pics/{name}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close('all')