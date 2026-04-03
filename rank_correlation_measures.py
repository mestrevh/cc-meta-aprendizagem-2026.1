from scipy.stats import weightedtau, kendalltau, spearmanr
import numpy as np

def weightedrankcorrelation(rankA, rankB):
    numerator = 0
    n = len(rankA)
    denominator = n**4 + n**3 - n**2 - n
    if denominator == 0: 
        return np.nan
    for r, q in zip(rankA, rankB):
        numerator += ((r - q) ** 2)*((n - r + 1)+(n - q + 1))
    rw = 1 - 6*numerator / denominator 
    return rw


rankA = [1,2,3,4,5,6]
rankB = [1,2,3,4,5,6]
rankC = [2,1,3,4,5,6]
rankD = [1,2,3,4,6,5]
rankE = [6,5,4,3,2,1]
pred_ranks = {'B':rankB, 'C':rankC, 'D':rankD, 'E':rankE}
for k in pred_ranks:
    print(f'Spearman correlation between A and {k}: {spearmanr(rankA, pred_ranks[k]).statistic:.2f}')
    print(f'Kendall Tau correlation between A and {k}: {kendalltau(rankA, pred_ranks[k]).statistic:.2f}')
    print(f'Weighted Tau correlation between A and {k}: {weightedtau(rankA, pred_ranks[k]).statistic:.2f}')
    print(f'Weighted Rank Correlation between A and {k}: {weightedrankcorrelation(rankA, pred_ranks[k]):.2f}')
    print('---')



