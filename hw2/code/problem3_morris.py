import numpy as np
import pandas as pd
import re
from collections import Counter
from math import log2


with open('Jack.txt', 'r') as fp:
    txt = fp.read().lower()
regex = re.compile('[^a-z ]')
txt = regex.sub('', txt)

word_freqs = pd.value_counts(txt.split()).to_numpy()
word_freqs = word_freqs / float(word_freqs.sum())
H = - np.sum(word_freqs * np.log2(word_freqs))
print(f'Entropy of words in jack:\nH(w) = {H}\n\n')

χ = list(txt)
lgth = len(txt)
letter_freqs = Counter(χ)
bigrams = Counter(zip(χ[:-1], χ[1:]))

Hc = 0
for (x1, x2), c in bigrams.items():
    Hc -= c / (lgth + 1) * (log2(letter_freqs[x1] / lgth) + log2(letter_freqs[x2] / lgth))
print(f'Cross-Entropy of Bigram and idependent single-letter dists:\nH_C(P_XY,P_X·P_Y) = {Hc}')
