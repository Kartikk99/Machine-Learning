import pandas as pd
import numpy as np
from Gradient_Descent_implementatin import gradient_descent
from sklearn import linear_model


def sklearn_res(df_pass):
    rg = linear_model.LinearRegression()
    rg.fit(df_pass[['math']], df_pass.cs)
    return rg.coef_, rg.intercept_


df = pd.read_csv(r"F:\Machine Learning\ML\3_gradient_descent\Exercise\test_scores.csv")

x = np.array(df['math'])
y = np.array(df['cs'])

gradient_descent(x, y, lr=0.0001, iterate=1000000)
m, b = sklearn_res(df)
# print('Gradient Descent: m={}, b={}'.format(m_grad, b_grad))
print('SkLearn Coefficient: m= {}, b= {}'.format(m, b))
