# This script simulates random sequences of transitions to study the properties of various
# transition metrics. It should be easy to modify should you wish to try different sequence lengths,
# numbers of states, etc.
#
# The loop_removal() function requires Graphviz to be installed so that "dot" can be run.
from subprocess import call

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import transition_metrics


SIMULATIONS = 10000  # Takes a couple hours to run everything
plt.style.use('nigel.mplstyle')
np.random.seed(11798)  # Catch 22


def simulate(seq_len, base_rates, remove_loops, verbose=True):
    """Generate a number of random sequences of states with the given base rates, calculate
    transition metrics for those sequences, and summarize the results.

    Args:
        seq_len (int): Number of elements to generate in each sequence
        base_rates (dict): Mapping of {state name: base rate (as a proportion)}
        verbose (bool, optional): Whether or not to print progress

    Returns:
        tuple: ({metric name: {state a: {state b: list of metric values}}},
                {state name: base rate after any loop removal})
    """
    pool = np.concatenate([[s] * int(f * 1000) for s, f in base_rates.items()])
    result = {}
    post_base_rates = {s: [] for s in base_rates} if remove_loops else base_rates
    for sim_i in tqdm(range(SIMULATIONS), desc='  Simulating', disable=not verbose):
        seq = np.random.choice(pool, seq_len)
        for metric_name, metric_values in [
                ('MCM', transition_metrics.calc_mcm(seq, remove_loops)),
                ('L', transition_metrics.calc_l(seq, remove_loops, False)),
                ('Lambda', transition_metrics.calc_l(seq, remove_loops, True)),
                # ('Phi', transition_metrics.calc_phi(seq, remove_loops)),
                ('LSA', transition_metrics.calc_lsa(seq, remove_loops)),
                ('LSA-5', transition_metrics.calc_lsa(seq, remove_loops, 5)),
                # ('Kappa', transition_metrics.calc_kappa(seq, remove_loops)),
                ('Q', transition_metrics.calc_yules_q(seq, remove_loops))]:
            if metric_name not in result:
                result[metric_name] = {a: {b: [] for b in base_rates} for a in base_rates}
            for a in base_rates:
                for b in base_rates:
                    try:
                        result[metric_name][a][b].append(metric_values[a][b])
                    except KeyError:
                        result[metric_name][a][b].append(np.nan)
        if remove_loops:
            seq = transition_metrics.get_without_loops(seq)
            for state in base_rates:
                post_base_rates[state].append((seq == state).mean())
    for state in base_rates:
        post_base_rates[state] = np.mean(post_base_rates[state])
    return result, post_base_rates


def nans_vs_seqlen(min_seq_len=5, max_seq_len=25, state_counts=[2, 4]):
    # Graph NaNs per sequence length for 2 example numbers of states (2 and 4)
    assert min_seq_len % 5 == 0 and max_seq_len % 5 == 0  # For X-axis graphing assumptions
    for num_states in state_counts:
        base_rates = {'s' + str(s): 1 / num_states for s in range(num_states)}
        nanprop = {}  # metric: [prop NaNs (mean across transitions)]
        for seq_len in range(min_seq_len, max_seq_len + 1):
            print('\nstates = ' + str(num_states) + '\tseq len = ' + str(seq_len))
            res, _ = simulate(seq_len, base_rates, False)
            for metric in set(res) - set(['Lambda']):
                if metric not in nanprop:
                    nanprop[metric] = []
                props = []
                for a in res[metric]:
                    for b in res[metric][a]:
                        props.append(np.isnan(res[metric][a][b]).mean())
                nanprop[metric].append(np.mean(props))

        line_options = {
            # 'Kappa': {},
            'L': {'linewidth': 4},
            'LSA': {'linewidth': 6},
            'MCM': {'linestyle': '--'},
            # 'Phi': {'linewidth': 3, 'linestyle': '--'},
            'Q': {},
            'LSA-5': {},
        }
        plt.figure(figsize=[6, 4.5])
        for metric in sorted(nanprop.keys()):
            if metric == 'LSA-5':
                plt.plot(range(min_seq_len + 5, max_seq_len + 1), nanprop[metric][5:], label=metric,
                         **line_options[metric])
            else:
                plt.plot(range(min_seq_len, max_seq_len + 1), nanprop[metric], label=metric,
                         **line_options[metric])
        plt.xticks(range(min_seq_len, max_seq_len + 1, 5))
        plt.xlabel('Sequence length')
        plt.ylabel('Prop. invalid values')
        plt.legend(loc='upper right')
        plt.savefig('graphs/nans_vs_seqlen_' + str(num_states) + 'states.svg')
        plt.close()


def maxmin_vs_seqlen(min_seq_len=5, max_seq_len=25, state_counts=[2, 4]):
    # Graph of max and min value per sequence length (2 and 4 states examples).
    # This is the max (or min) of all transitions, not the max of all simulations, which would
    # just be 1 because of outliers.
    assert min_seq_len % 5 == 0 and max_seq_len % 5 == 0  # For X-axis graphing assumptions
    for num_states in state_counts:
        base_rates = {'s' + str(s): 1 / num_states for s in range(num_states)}
        min_transition = {}
        min_std = {}
        max_transition = {}
        max_std = {}
        for seq_len in range(min_seq_len, max_seq_len + 1):
            print('\nstates = ' + str(num_states) + '\tseq len = ' + str(seq_len))
            res, _ = simulate(seq_len, base_rates, False)
            for metric in set(res) - set(['Lambda']):
                if metric not in min_transition:
                    min_transition[metric] = []
                    min_std[metric] = []
                    max_transition[metric] = []
                    max_std[metric] = []
                vals = []
                stds = []
                for a in res[metric]:
                    for b in res[metric][a]:
                        vals.append(np.nanmean(res[metric][a][b]))
                        stds.append(np.nanstd(res[metric][a][b]))
                min_transition[metric].append(np.nanmin(vals))
                min_std[metric].append(stds[np.argmin(vals)])
                max_transition[metric].append(np.nanmax(vals))
                max_std[metric].append(stds[np.argmax(vals)])

        plt.figure(figsize=[6, 4.5])
        ci = 1.96 / np.sqrt(SIMULATIONS)
        plot_min, plot_max = 1, 0
        for c, metric in enumerate(sorted(min_transition.keys())):
            plt.plot(range(min_seq_len, max_seq_len + 1), max_transition[metric],
                     label=metric, color='C' + str(c))
            plt.fill_between(range(min_seq_len, max_seq_len + 1),
                             np.array(max_transition[metric]) - ci * np.array(max_std[metric]),
                             np.array(max_transition[metric]) + ci * np.array(max_std[metric]),
                             color='C' + str(c), alpha=.1)
            plot_min = min(plot_min, min(max_transition[metric]))
            plot_max = max(plot_max, max(max_transition[metric]))
        plt.plot(range(min_seq_len, max_seq_len + 1), [0] * len(max_transition[metric]),
                 color='#000000', alpha=.4, linestyle='--')
        plt.xticks(range(min_seq_len, max_seq_len + 1, 10))
        plt.ylim([plot_min - .1, plot_max + .1])
        plt.xlabel('Sequence length')
        plt.ylabel('Values of transition metrics')
        plt.legend(loc='upper right')
        plt.savefig('graphs/maxmin_vs_seqlen_' + str(num_states) + 'states.svg')
        plt.close()


def loop_removal(seq_len=100):
    # Graph the effect of loop removal for various metrics
    base_rates = {'S1:': .25, 'S2:': .25, 'S3:': .50}
    res, pbr = simulate(seq_len, base_rates, remove_loops=True)
    for metric in res:
        print('Creating state transition diagram for ' + metric)
        # Relabel states with post-transition-removal base rate (PBR)
        for a in list(res[metric]):
            for b in list(res[metric][a]):
                res[metric][a][b + ' %.0f%%' % (100 * pbr[b])] = res[metric][a].pop(b)
            res[metric][a + ' %.0f%%' % (100 * pbr[a])] = res[metric].pop(a)
        # Save graph
        with open('graphs/tmp.dot', 'w') as outfile:
            outfile.write(transition_metrics.transitions_to_dot(res[metric]) + '\n')
        call(['dot', 'graphs/tmp.dot', '-Tpng', '-Gdpi=300', '-o',
              'graphs/states_self_removed_' + metric + '.png'])

    # Without loop removal
    base_rates = {'S1: 25%': .25, 'S2: 25%': .25, 'S3: 50%': .50}
    res, _ = simulate(seq_len, base_rates, remove_loops=False)
    for metric in res:
        print('Creating state transition diagram for ' + metric)
        with open('graphs/tmp.dot', 'w') as outfile:
            outfile.write(transition_metrics.transitions_to_dot(res[metric]) + '\n')
        call(['dot', 'graphs/tmp.dot', '-Tpng', '-Gdpi=300', '-o',
              'graphs/states_self_unmodified_' + metric + '.png'])


if __name__ == '__main__':
    # loop_removal()
    nans_vs_seqlen()
    maxmin_vs_seqlen(max_seq_len=100)
    maxmin_vs_seqlen(max_seq_len=100, state_counts=[7])  # Match Baker et al. 2007
    nans_vs_seqlen(max_seq_len=50, state_counts=[7])
