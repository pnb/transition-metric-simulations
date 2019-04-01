# Functions to calculate various state transition metrics. This file is stand-alone and has few
# dependencies, so it should be straightforward to include in other projects.
import sys
import numpy as np


assert 1 / 2 > 0, 'This script only works with Python 3'


def _iterate_state_pairs(state_list, remove_loops=False):
    state_list = np.array(state_list)
    if remove_loops:
        state_list = get_without_loops(state_list)
    seq_prev = state_list[:-1]
    seq_next = state_list[1:]
    for a in np.unique(state_list):
        arr1 = seq_prev == a
        for b in np.unique(state_list):
            arr2 = seq_next == b
            yield a, b, arr1, arr2


def get_without_loops(state_list):
    return np.array([state_list[0]] + [state_list[i] for i in range(1, len(state_list))
                    if state_list[i - 1] != state_list[i]])


def calc_l(state_list, remove_loops=False, removal_correction=False):
    """Calculate D'Mello's L, with some corrections for edge cases, including the last state in the
    sequence and the case where loops (self-transitions) are not allowed.

    See: D’Mello, S., Taylor, R. S., & Graesser, A. (2007). Monitoring affective trajectories during
    complex learning. In Proceedings of the 29th annual meeting of the cognitive science society
    (pp. 203–208). Austin, TX: Cognitive Science Society.

    Args:
        state_list (iterable): State sequence (e.g., list of strings)
        remove_loops (bool, optional): Remove loops (state persistence)
        removal_correction (bool, optional): If removing loops, correct L for increased
            probability due to the impossibility of self-transitions (i.e., compute lambda)

    Returns:
        dict: Mapping of {state A: {state B: transition L}} for all state pairs
    """
    result = {a: {} for a in np.unique(state_list)}
    orig_next = np.array(state_list)[1:]
    for a, b, seq_prev, seq_next in _iterate_state_pairs(state_list, remove_loops):
        if sum(seq_prev) > 0:
            observed = sum(seq_prev & seq_next) / sum(seq_prev)  # Markov chain model probability.
        else:
            observed = np.nan
        if remove_loops and removal_correction:
            # Base rate of next state in original data after removing cur state.
            expected = sum(orig_next == b) / (len(orig_next) - sum(orig_next == a))
        else:
            expected = np.mean(seq_next)
        result[a][b] = (observed - expected) / (1 - expected) if expected < 1 else np.nan
    return result


def calc_lambda(state_list, remove_loops=False):
    """Calculate lambda, which corrects for loop removal if needed, so that chance level remains 0.
    Maximum lambda is 1, chance level is 0, and minimum is -infinity.

    Args:
        state_list (iterable): State sequence (e.g., list of strings)
        remove_loops (bool, optional): Remove loops (state persistence)

    Returns:
        dict: Mapping of {state A: {state B: transition lambda}} for all state pairs
    """
    return calc_l(state_list, remove_loops, removal_correction=True)


def calc_phi(state_list, remove_loops=False):
    """Calculate transition likelihood according to the phi correlation between occurrences of
    previous and next states. This is equivalent to Pearson's r for binary variables, and to the
    Matthews correlation coefficient (MCC).

    Args:
        state_list (iterable): State sequence of any hashable type (e.g., list of strings)
        remove_loops (bool, optional): Remove loops (state persistence)

    Returns:
        dict: Mapping of {state A: {state B: transition phi}} for all state pairs
    """
    result = {a: {} for a in np.unique(state_list)}
    for a, b, seq_prev, seq_next in _iterate_state_pairs(state_list, remove_loops):
        result[a][b] = np.corrcoef(seq_prev, seq_next)[1, 0]
    return result


def calc_lsa(state_list, remove_loops=False):
    """Calculate z-scores of transition likelihoods, often referred to as Lag Sequential Analysis.

    Args:
        state_list (iterable): State sequence of any hashable type (e.g., list of strings)
        remove_loops (bool, optional): Remove loops (state persistence)

    Returns:
        dict: Mapping of {state A: {state B: transition z-score}} for all state pairs
    """
    # "Observing interaction: An introduction to sequential analysis"
    # "Lag Sequential Analysis in Social Work Research and Practice"
    # Formula also found in code from "Uncovering what matters: Analyzing transitional relations
    #   among contribution types in knowledge-building discourse"
    # "Testing Sequential Association: Estimating Exact p Values Using Sampled Permutations"
    #
    # Zi->j = (Oi->j - Ei->j) / sqrt(Ei->j * (1 - Xi+ / N) * (1 - X+j / N))
    # Where:
    #   Zi->j is a z-score for the transition likelihood.
    #   N is the total number of transitions
    #   Oi->j is observed count of i -> j transitions
    #   Ei->j is expected count of i -> j transitions
    #   Xi+ is count of transitions FROM i
    #   X+j is count of transitions TO j
    z = {a: {} for a in np.unique(state_list)}
    for i, j, seq_prev, seq_next in _iterate_state_pairs(state_list, remove_loops):
        N = len(seq_prev)
        N = N if N > 0 else np.nan
        oij = sum(seq_prev & seq_next)
        eij = sum(seq_prev) * sum(seq_next) / N
        xip = sum(seq_prev)
        xpj = sum(seq_next)
        z[i][j] = (oij - eij) / np.sqrt(eij * (1 - xip / N) * (1 - xpj / N))
    return z


def calc_kappa(state_list, remove_loops=False):
    """Calculate Cohen's kappa as a transition metric, counting each consecutive pair of states in
    the sequence as agreement for whatever two states those are, and disagreement for all others.

    Args:
        state_list (iterable): State sequence of any hashable type (e.g., list of strings)
        remove_loops (bool, optional): Remove loops (state persistence)

    Returns:
        dict: Mapping of {state A: {state B: transition kappa}} for all state pairs
    """
    result = {a: {} for a in np.unique(state_list)}
    for a, b, seq_prev, seq_next in _iterate_state_pairs(state_list, remove_loops):
        n = len(seq_prev)
        n = n if n > 0 else np.nan
        acc = sum(seq_prev == seq_next) / n
        exp_acc = (sum(seq_prev) * sum(seq_next) / n +
                   sum(np.invert(seq_prev)) * sum(np.invert(seq_next)) / n) / n
        result[a][b] = (acc - exp_acc) / (1 - exp_acc)
    return result


def calc_yules_q(state_list, remove_loops=False):
    """Calculate Yule's Q, a rank correlation measure with no adjustment for ties. Thus, increasing
    the length of a sequence does not neccessarily change Q, which can cause bias.

    Args:
        state_list (iterable): State sequence of any hashable type (e.g., list of strings)
        remove_loops (bool, optional): Remove loops (state persistence)

    Returns:
        dict: Mapping of {state A: {state B: transition Q}} for all state pairs
    """
    result = {a: {} for a in np.unique(state_list)}
    for a, b, seq_prev, seq_next in _iterate_state_pairs(state_list, remove_loops):
        ad = sum(seq_prev & seq_next) * sum(np.invert(seq_prev) & np.invert(seq_next))
        bc = sum(seq_prev & np.invert(seq_next)) * sum(np.invert(seq_prev) & seq_next)
        result[a][b] = (ad - bc) / (ad + bc) if ad + bc > 0 else np.nan
    return result


def calc_mcm(state_list, remove_loops=False):
    """Markov chain model, i.e. a model with transitions between each state according to simple
    probability unadjusted for base rates. This method is thus heavily influenced by base rates.

    Args:
        state_list (iterable): State sequence of any hashable type (e.g., list of strings)
        remove_loops (bool, optional): Remove loops (state persistence)

    Returns:
        dict: Mapping of {state A: {state B: transition probability}} for all state pairs
    """
    result = {a: {} for a in np.unique(state_list)}
    for a, b, seq_prev, seq_next in _iterate_state_pairs(state_list, remove_loops):
        result[a][b] = sum(seq_prev & seq_next) / sum(seq_prev) if sum(seq_prev) > 0 else np.nan
    return result


def transitions_to_dot(transition_map, abs_min_cutoff=0.):
    """Create Graphviz DOT syntax from an adjacency list of transitions (e.g., phi values). If
    values are iterable, means and standard deviations will be calculated.

    Args:
        transition_map (dict): Adjacency list with transition values, e.g. {'a': {'b': .5}}
        abs_min_cutoff (float, optional): Do not include edges with |value| < cutoff

    Returns:
        string: DOT syntax that can be saved to a file and/or displayed with Graphviz
    """
    all_states = set(transition_map)
    for a in transition_map:
        all_states.update(transition_map[a])
    g = '\n  ' + ';\n  '.join('"' + str(s) + '"' for s in sorted(all_states))  # Suggest node order
    nans = 0
    count = 0
    for a in transition_map:
        for b in transition_map:
            nans += np.sum(np.isnan(transition_map[a][b]))
            count += np.array(transition_map[a][b]).size
            m = np.nanmean(transition_map[a][b])
            sd = np.nanstd(transition_map[a][b])
            if abs(m) >= abs_min_cutoff:
                g += ';\n  "%s" -> "%s" [label=" %.3f (%.3f)"]' % (a, b, m, sd)
    return 'digraph {\n  node[width=1, shape=circle];' + \
           '\n  labelloc="t";\n  label="Prop. NaNs = %.6f";%s;\n}' % (nans / count, g)


if __name__ == '__main__':
    # Testing examples
    print(calc_l([1, 1, 2, 2]))
    print(calc_l([1, 1, 2, 2, 1, 3, 2, 2], remove_loops=True))
    print(calc_l([1, 2, 2, 3, 2, 1, 2, 2, 3, 1, 2, 3, 2, 3, 1, 2], remove_loops=True))
    print(calc_lsa([1, 1, 2, 2]))
    print(calc_phi([1, 1, 2, 2]))
    print(calc_phi([1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    print(calc_yules_q([1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    print(calc_mcm([1, 2, 1, 1]))
    print(calc_phi([1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    print(calc_kappa([1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
