
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
'''
データのフォーマット
例えば、10個のモデルのチェックを多なった結果を以下のように表す
data = np.array([
    [1, 1],
    [1, 0],
    [1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
])
左の列: Symptomの有無(1 -> 有)
右の列: Causeの有無(1 -> 有)

上記のデータの場合、
(Symptom有, Cause有)が1件
(Symptom有, Cause無)が2件
(Symptom無, Cause無)が7件

存在したということを意味する
'''

def get_logistic_regression_coef(num_unit, num_symptom, num_cause):
    lr = LogisticRegression()
    data = np.array(
        [[0, 0]] * (num_unit - num_symptom)
        + [[1, 0]] * (num_symptom - num_cause)
        + [[1, 1]] * num_cause
    )
    X = np.expand_dims(data[:, 0], 1)
    y = data[:, 1]
    lr.fit(X, y)

    params = np.append(lr.intercept_, lr.coef_)
    predictions = lr.predict(X)

    newX = np.append(np.ones((len(X), 1)), X, axis=1)
    MSE = (sum((y - predictions) ** 2) / (len(newX) - len(newX[0])))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2*(1-stats.t.cdf(np.abs(i), (len(newX)-1))) for i in ts_b]
    print(num_unit, num_symptom, num_cause, lr.coef_[0][0], p_values[1])
    return lr.coef_[0][0], p_values[1]

if __name__ == '__main__':
    print(get_logistic_regression_coef(10, 3, 2))
