from .Wishart1 import Wishart
import numpy as np
import warnings

from tqdm import tqdm


def get_motifs(pattern_set, data, do_clustering=False, r=11, mu=0.2, beta=1, le=3):
    def cluster_motifs(samples, r, mu):
        clusters = [Wishart(r, mu).fit(i) for i in tqdm(samples)]
        motifs = []
        for n, samp in enumerate(samples):
            motifs.append([])
            for i in range(max(clusters[n]) + 1):
                motifs[-1].append(np.mean(samp[np.where(clusters[n] == i)], axis=0))
            motifs[-1] = np.array(motifs[-1])

        return motifs

    def get_sample_for_a_pattern(a, le, pat=None):
        if pat is None:
            pat = np.int_(np.ones(le))
        b = np.int_(np.append([0], np.cumsum(pat)))
        l = []
        for i in range(a.shape[0] - np.sum(pat)):
            l.append(a[b].reshape(1, -1))
            b = b + 1

        return np.concatenate(l, axis=0)

    warnings.filterwarnings("ignore")
    s_s_len = len(pattern_set)

    if beta == 1:
        samples_set = np.array([get_sample_for_a_pattern(data, le, pat=pattern) for pattern in pattern_set])
    elif beta < 1:
        chosen_pats_samples = np.random.choice(np.arange(s_s_len), size=int(s_s_len * beta), replace=False)
        samples_set = np.array(
            [get_sample_for_a_pattern(data, le, pat=pattern) for pattern in np.array(pattern_set)[chosen_pats_samples]])
    # print(cluster_motifs)
    if do_clustering:
        samples_set = cluster_motifs(samples_set, r, mu)

    return samples_set
