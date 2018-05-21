
import numpy as np
from sklearn.linear_model import LogisticRegression
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

def get_logistic_regression_coef(num_model, num_symptom, num_cause):
    print(num_model, num_symptom, num_cause)
    lr = LogisticRegression()
    data = np.array(
        [[0, 0]] * (num_model - num_symptom)
        + [[1, 0]] * (num_symptom - num_cause)
        + [[1, 1]] * num_cause
    )
    lr.fit(np.expand_dims(data[:, 0], 1), data[:, 1])
    return lr.coef_[0][0]

if __name__ == '__main__':
    print(get_logistic_regression_coef(10, 3, 2))
    print(get_logistic_regression_coef(10, 9, 9))
