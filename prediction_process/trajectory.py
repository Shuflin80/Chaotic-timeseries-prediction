import numpy as np

from tqdm import tqdm
from scipy.spatial import distance
from dbscan1d import DBSCAN1D


class TrajectoryPrediction:
    def __init__(self, pattern_set, x, samples_set, y_true, eps=0.005, sigma=0.01, tresh_size1=0.04, tresh_diff=0.2, eps_cl=0.01, min_samples_cl=5):
        # remove
        self.pattern_set = pattern_set
        self.samples_set = samples_set
        self.x = x
        self.tresh_diff = tresh_diff
        self.eps = eps
        self.sigma = sigma
        self.motifs = None
        # the point is deemed unpredictable if the largest cluster comprizes < 4% of all prediction points
        self.tresh_size1 = tresh_size1
        self.eps_cl = eps_cl
        self.min_samples_cl = min_samples_cl
        self.y = self.x.copy()
        self.y_true = y_true

    def possible_predictions(self, pattern, x, motifs):
        def C(pattern, x):
            c_ = len(x) - np.cumsum(pattern[::-1])
            return x[c_][::-1]

        C_ = C(pattern, x).tolist()
        C_ = [np.float32(i) if i != 'N' else 'N' for i in C_]
        TrCa = motifs[: ,:-1]
        if 'N' in C_:

            return []
        else :
            d = distance.cdist(TrCa, [C_], 'euclidean')
            closed_mot_index = np.where(d < self.eps)[0]
            Possible_predictions_ = motifs[closed_mot_index, -1]

            return Possible_predictions_

    def get_possible_prediction_values(self, y):

        Qlist = np.array([self.possible_predictions(self.pattern_set[i], y, self.samples_set[i]) for i in range(len(self.pattern_set))])
        Q = np.hstack(Qlist)
        if len(Q) == 0:
            return 'N'
        else:
            return Q

    def predict_one_trajpoint_dbscan(self, y, rand_pert, many):
        Q = self.get_possible_prediction_values(y)
        if len(Q) == 0:
            return 'N'
        elif Q == 'N':
            return 'N'
        Qreshape = Q.reshape(len(Q) ,1)
        c = DBSCAN1D(eps=self.eps_cl, min_samples=self.min_samples_cl)
        l = c.fit_predict(Qreshape)
        count_cl = np.array(np.unique(l, return_counts = True)).T

        count_cl_without_noize = count_cl[~np.isin(count_cl[: ,0] ,[-1])]
        if count_cl_without_noize.shape[0] == 0:

            return 'N'

        elif many:
            max_cl_n = count_cl_without_noize[np.argsort(count_cl_without_noize[:, 1]), :]
            return np.mean(Q[np.where(1 == max_cl_n[-1, 0])]) + np.random.normal(0, self.sigma, 1)[0]

        else:
            count_cl_without_noize_sorted = count_cl_without_noize[np.argsort(count_cl_without_noize[: ,1]), :]
            # size1 - size of the largest cluster
            size1 = count_cl_without_noize_sorted[-1, 1] / Q.size
            if count_cl_without_noize_sorted.shape[0] > 1:
                # size2 - size of the second-largest cluster
                size2 = count_cl_without_noize_sorted[-2, 1] / Q.size
            else:
                size2 = 0
            if size2 != 0:
                if (size1 < self.tresh_size1) | (size1 - size2 < self.tresh_diff):
                    return 'N'
                else:
                    if rand_pert:
                        return np.mean(Q[np.where(l == count_cl_without_noize_sorted[-1, 0])]) + np.random.normal(0, self.sigma, 1)[0]
                    else:
                        return np.mean(Q[np.where(l == count_cl_without_noize_sorted[-1, 0])])
            else:
                if size1 < self.tresh_size1:
                    return 'N'
                else:
                    if rand_pert:
                        return np.mean(Q[np.where(l == count_cl_without_noize_sorted[-1, 0])]) + np.random.normal(0, self.sigma, 1)[0]
                    else:
                        return np.mean(Q[np.where(l == count_cl_without_noize_sorted[-1, 0])])

    def My_pull(self, T, rand, many, y=None):
        if y is None:
            y = self.x.copy()

        pred = []
        for i in range(T):
            pred.append(self.predict_one_trajpoint_dbscan(y, rand, many))
            y = np.append(y, pred[-1])
        return pred


    def pull_ppvs(self, T, step, samples):
        y = self.x.copy()
        all_preds = []
        for loop in tqdm(range(samples)):

            pred = []
            for i in range(T):
                pred.append(self.get_possible_prediction_values(y))
                y = np.append(y, pred[-1])
            y = np.append(self.x, self.y_true[:(loop +1 ) *step])
            all_preds.append(pred)
        return all_preds

    def predict(self, h, sample_size, noise: bool, step=1, s=1):
        self.preds = []

        if s > 1:
            many = True
        else:
            many = False

        for i in range(s):
            self.preds.append([self.My_pull(h, noise, many, np.append(self.x, self.y_true[:step *j])) for j in tqdm(range(sample_size))])
        return self.preds