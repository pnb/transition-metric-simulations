# Simulate very long sequences once to see that LSA indeed goes toward 0.
import numpy as np

from simulate import simulate


for seqlen in [100, 1000, 2000, 3000]:
    print('\nSequences of length', seqlen)
    res, _ = simulate(seqlen, {'A': .5, 'B': .5}, remove_loops=False, verbose=True)
    for a in res['LSA']:
        for b in res['LSA'][a]:
            print(a, '=>', b, 'mean LSA =', np.nanmean(res['LSA'][a][b]))
