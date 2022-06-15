import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error


def parameters_function(p, tr, h):
    ###########
    ### k goes from 1 to 10
    def get_h_p(p, k):
        v = []
        for i in range(len(p)):
            try:
                v.append(p[i][k - 1])
            except IndexError:
                pass
        return v

    #############

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    #############
    pred_all = [list(zip(get_h_p(tr, i), get_h_p(p, i))) for i in range(1, h + 1)]
    rmse_dem = []
    mape_dem = []
    N_score = []
    for i in range(len(pred_all)):
        metric = []
        N_ = 0
        for y in pred_all[i]:
            if 'N' in y:
                N_ += 1
            else:
                metric.append([y[0], y[1]])
        N_ = N_ / len(pred_all[i])
        metric = np.array(metric)
        if metric.size > 0:
            if len(metric) > 1:
                score_rmse = mean_squared_error(np.float_(metric[:, 0]), np.float_(metric[:, 1]), squared=False)
                score_mape = mean_absolute_percentage_error(np.float_(metric[:, 0]), np.float_(metric[:, 1]))
                rmse_dem.append(score_rmse)
                mape_dem.append(score_mape)
                N_score.append(N_)
        else:
            rmse_dem.append(np.nan)
            mape_dem.append(np.nan)
            N_score.append(1)

    return rmse_dem, mape_dem, N_score, pred_all


def plot_metrics(params, name):
    plt.figure(figsize=(10, 6))
    plt.grid(visible=True, which='both', axis='both', alpha=0.5)
    plt.plot(params[0])
    plt.title(f'{name} \n' + 'rmse sum' + ' ' + str(f'{sum(params[0]):.4f}'))
    plt.show()
    print(sum(params[0]))

    plt.figure(figsize=(10, 6))
    plt.grid(visible=True, which='both', axis='both', alpha=0.5)
    plt.plot(params[1])
    plt.title(f'{name} \n' + 'mape sum' + ' ' + str(f'{sum(params[1]):.4f}'))
    plt.show()
    print(sum(params[1]))

    plt.figure(figsize=(10, 6))
    plt.grid(visible=True, which='both', axis='both', alpha=0.5)
    plt.plot(params[2])
    plt.title(f'{name} \n' + 'N% sum' + ' ' + str(f'{sum(params[2]):.4f}'))
    plt.show()
    print(sum(params[2]))


def plot_predictions(preds, true, title, size):
    f_preds = np.ravel(preds)
    plt.figure(figsize=(20, 10))
    plt.plot(range(size), [np.float64(i) if i != 'N' else np.nan for i in f_preds], color='r', label='predicted_values')
    plt.plot(true[:size], alpha=0.7, label='true_values')
    plt.legend(fontsize=12)
    plt.title(title, fontsize=20)
