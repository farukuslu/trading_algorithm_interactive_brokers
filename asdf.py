from hw2_utils import *
from sklearn import linear_model
from sklearn.metrics import r2_score
from pandas import *

bonds_data = DataFrame(fetch_usdt_rates(2021))

def yolo(yields_row):
    maturities = DataFrame([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2])
    linreg_model = linear_model.LinearRegression()
    linreg_model.fit(maturities, yields_row)
    modeled_bond_rates = linreg_model.predict(maturities)
    return [linreg_model.coef_[0], linreg_model.intercept_, r2_score(yields_row,
                                                                  modeled_bond_rates)]

bonds_features = bonds_data[["1 mo", "2 mo", "3 mo", "6 mo", "1 yr",
                          "2 yr"]].apply(yolo, axis=1,
                                         result_type='expand')

bonds_features.columns = ["a", "b", "R2"]

yields_row = bonds_data.loc[1,["1 mo", "2 mo", "3 mo", "6 mo", "1 yr", "2 yr"]]

asdf = zip(
    *
)


