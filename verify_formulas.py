# Metrics in the paper are expressed in terms of contingency table cells, but actual calculations
# use mostly NumPy methods to make faster/more succinct calculations. Here we double-check the
# contingency table versions of the formulas to make sure they match the main implementations.
import transition_metrics
import pandas as pd
import numpy as np


seq_i = np.array(list('XYXXZXYXYZYZ'))
# pd.crosstab order:
#   D   C
#   B   A
#   D = False False, C = True False, B = False True, A = True True
lam = transition_metrics.calc_lambda(seq_i, remove_loops=True)

# Check the X -> Y transition
seq_r = transition_metrics.get_without_loops(seq_i)
print(seq_i)
print(seq_r)
print(lam['X']['Y'])
ct_i = pd.crosstab(pd.Series(seq_i[:-1] == 'X'), pd.Series(seq_i[1:] == 'Y'))
ct_r = pd.crosstab(pd.Series(seq_r[:-1] == 'X'), pd.Series(seq_r[1:] == 'Y'))
A = ct_i[True][True]
B = ct_i[False][True]
C = ct_i[True][False]
D = ct_i[False][False]

Ar = ct_r[True][True]
Br = ct_r[False][True]

k = sum(seq_i[1:] == 'X')

ct_lambda = (Ar / (Ar + Br) - (A + C) / (A + B + C + D - k)) / (1 - (A + C) / (A + B + C + D - k))
print(ct_lambda)
