import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


plt.style.use('nigel.mplstyle')


df = pd.read_csv('uci_epm-results.csv')
student_level = df.groupby('student_id').mean()
seq_level = df.groupby('sequence_length').mean()
seq_level_maxlen = seq_level.loc[['all']]
seq_level = seq_level[seq_level.index != 'all']
seq_level.index = seq_level.index.map(float)
seq_level.sort_index(inplace=True)

print('Prop. decrease in sequence length with loop removal:')
noloop_diff = student_level.orig_sequence_length - student_level.activity_name_noloop_length
print('Activity name =', (noloop_diff / student_level.orig_sequence_length).mean())
noloop_diff = student_level.orig_sequence_length - student_level.exercise_noloop_length
print('Exercise =', (noloop_diff / student_level.orig_sequence_length).mean())

metric_names = pd.unique([f.split('-mean-')[-1] for f in df if '-mean-' in f])
seq_names = pd.unique([f.split('-')[0] for f in df if '-mean-' in f])

# All metrics per sequence
for seq_name in tqdm(seq_names, 'Plotting all metrics'):
    for agg_type in ['min', 'max', 'mean']:
        plt.figure()
        for metric in metric_names:
            if metric != 'Lambda' or '_noloop' in seq_name:
                plt.plot(seq_level.index, seq_level[seq_name + '-' + agg_type + '-' + metric],
                         label=metric if metric != 'Lambda' else 'L*')
        plt.xlabel('Sequence length')
        plt.ylabel('Metric value')
        plt.legend()
        if agg_type == 'min':
            plt.title('Minimum-value transition')
        elif agg_type == 'max':
            plt.title('Maximum-value transition')
        elif agg_type == 'mean':
            plt.title('Mean of transitions')
        bottom_y_lim, _ = plt.ylim()
        plt.ylim(bottom=min(0, bottom_y_lim))
        plt.savefig('graphs/epm/' + seq_name + '-' + agg_type + '.svg')
        plt.close()
# It seems like loops force everything else positive in L

# Bar graphs for max-length sequences
for seq_name in tqdm(pd.unique(seq_names), 'Plotting max-length sequences'):
    plt.figure()
    for agg_i, agg_type in enumerate(['min', 'max', 'mean']):
        bar_sizes = []
        bar_ys = []
        labels = []
        for metric_i, metric in enumerate(metric_names):
            if metric == 'Lambda' and 'noloop' not in seq_name:
                continue  # No point in plotting this
            lbl = agg_type if metric_i == 0 else None
            bar_ys.append(-agg_i * (len(metric_names) + 1) - len(labels))
            bar_sizes.append(seq_level_maxlen[seq_name + '-' + agg_type + '-' + metric].values[0])
            labels.append(metric)
            if metric == 'Lambda':
                labels[-1] = 'L*'  # Rename for literature consistency
            if abs(bar_sizes[-1]) < .02:
                bar_sizes[-1] = .02  # Ensure a tiny sliver of bar is visible
            plt.barh(bar_ys[-1], bar_sizes[-1], .9, label=lbl, color='C' + str(agg_i))
        if sum(bar_sizes) > 0:
            xoff = min(bar_sizes + [0])
            align = 'right'
        else:
            xoff = max(bar_sizes + [0])
            align = 'left'
        for label, bar_y in zip(labels, bar_ys):
            plt.annotate(' ' + label + ' ', (xoff, bar_y - .2), horizontalalignment=align)
    # plt.title('Sequence type: ' + seq_name)
    plt.title('Full-length sequences for each participant')
    plt.xlabel('Value of metric')
    plt.yticks([])
    plt.margins(x=.01, y=.01)
    plt.legend()
    plt.savefig('graphs/epm/' + seq_name + '-maxlen.svg')
    plt.close()
