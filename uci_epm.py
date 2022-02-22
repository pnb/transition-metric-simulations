# Test transition metrics on the UCI Educational Process Mining (EPM) dataset
import os
import datetime
import time
from functools import partial

from tqdm import tqdm
import pandas as pd
import numpy as np

import transition_metrics


def time_fmt(val):
    day, hour = val.split(' ')
    day, month, year = day.split('.')
    hour, minute, second = hour.split(':')
    dt = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    return int(time.mktime(dt.timetuple()))


header = ['session_num', 'student_id', 'exercise_id', 'activity', 'start_time', 'end_time',
          'idle_time_ms', 'mouse_wheel', 'mouse_wheel_clicks', 'mouse_clicks_left',
          'mouse_clicks_right', 'mouse_movements', 'keystrokes']
base_dir = 'uci_epm_data/Data/Processes/'
results = []
for sesh in sorted(os.listdir(base_dir)):
    for fname in tqdm(sorted(os.listdir(os.path.join(base_dir, sesh))), desc=sesh):
        df = pd.read_csv(os.path.join(base_dir, sesh, fname), sep=', ', names=header,
                         engine='python', converters={'end_time': time_fmt, 'start_time': time_fmt})
        df['activity'] = df.activity.str.lower()
        df['activity_name'] = df.activity.str.replace(r'_.*', '', regex=True)
        exercise_noloop_len = len(transition_metrics.get_without_loops(df.exercise_id))
        activity_name_noloop_len = len(transition_metrics.get_without_loops(df.activity_name))
        for seq_len in [v for v in range(5, 51)] + ['all']:
            results.append(pd.Series({
                'student_id': df.student_id.iloc[0],
                'session_num': df.session_num.iloc[0],
                'orig_sequence_length': len(df),
                'sequence_length': seq_len,
                'activity_name_num_states': df.activity_name.nunique(),
                'exercise_num_states': df.exercise_id.nunique(),
                'activity_name_noloop_length': activity_name_noloop_len,
                'exercise_noloop_length': exercise_noloop_len,
            }))
            if seq_len == 'all':
                seq_len = None  # Get the entire sequence by slicing from beginning to end
            metadata_col_count = len(results[-1])
            for metric, func in [('MCM', transition_metrics.calc_mcm),
                                 ('L', transition_metrics.calc_l),
                                 ('Lambda', transition_metrics.calc_lambda),
                                 ('LSA', transition_metrics.calc_lsa),
                                 ('LSA-5', partial(transition_metrics.calc_lsa, lag=5)),
                                 ('Q', transition_metrics.calc_yules_q)]:
                res = func(df.activity_name.iloc[:seq_len])
                results[-1]['activity_name-mean-' + metric] = \
                    np.nanmean([res[a][b] for a in res for b in res])
                results[-1]['activity_name-min-' + metric] = \
                    np.nanmin([res[a][b] for a in res for b in res])
                results[-1]['activity_name-max-' + metric] = \
                    np.nanmax([res[a][b] for a in res for b in res])
                res = func(df.exercise_id.iloc[:seq_len])
                results[-1]['exercise-mean-' + metric] = \
                    np.nanmean([res[a][b] for a in res for b in res])
                results[-1]['exercise-min-' + metric] = \
                    np.nanmin([res[a][b] for a in res for b in res])
                results[-1]['exercise-max-' + metric] = \
                    np.nanmax([res[a][b] for a in res for b in res])
                res = func(df.exercise_id.iloc[:seq_len], True)
                results[-1]['exercise_noloop-mean-' + metric] = \
                    np.nanmean([res[a][b] for a in res for b in res if a != b])
                results[-1]['exercise_noloop-min-' + metric] = \
                    np.nanmin([np.nan] + [res[a][b] for a in res for b in res if a != b])
                results[-1]['exercise_noloop-max-' + metric] = \
                    np.nanmax([np.nan] + [res[a][b] for a in res for b in res if a != b])

result_df = pd.DataFrame.from_records(results)
result_df = result_df.reindex(list(result_df.columns)[:metadata_col_count] +
                              sorted(result_df.columns[metadata_col_count:]), axis=1)
result_df.to_csv('uci_epm-results.csv', index=False)
