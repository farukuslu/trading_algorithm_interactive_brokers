import pandas

def model(bonds_hist, ivv_hist):




    from hw2_utils import *
    from sklearn import linear_model
    from sklearn.metrics import r2_score
    from pandas import *
    import pickle

    # pickle.dump(bonds_hist, open("bonds_hist.p", "wb"))
    # pickle.dump(ivv_hist, open("ivv_hist.p", "wb"))

    bonds_hist = read_json(pickle.load(open("bonds_hist.p", "rb")))
    ivv_hist = read_json(pickle.load(open("ivv_hist.p", "rb")))

    def bonds_fun(yields_row):
        maturities = DataFrame([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2])
        linreg_model = linear_model.LinearRegression()
        linreg_model.fit(maturities, yields_row[1:])
        modeled_bond_rates = linreg_model.predict(maturities)
        return [yields_row["Date"].date(), linreg_model.coef_[0],
                linreg_model.intercept_,
                r2_score(yields_row[1:],
                         modeled_bond_rates)]

    bonds_features = bonds_hist[["Date", "1 mo", "2 mo", "3 mo", "6 mo", "1 yr",
                                 "2 yr"]].apply(bonds_fun, axis=1,
                                                result_type='expand')

    bonds_features.columns = ["Date", "a", "b", "R2"]

    ivv_response = []

    for i in range(1, len(ivv_hist)):
        ivv_response.append(
            int(ivv_hist["High"][i] > ivv_hist["Close"][i-1] * 1.02)
        )

    bonds_features = bonds_features[1:]
    bonds_features["response"] = ivv_response
    bonds_features.set_index(list(range(0, len(bonds_features) - 1)))

    return bonds_features.to_json()
