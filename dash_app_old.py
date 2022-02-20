import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle
from datetime import date, datetime
from time import sleep
from momentum import *  # import everything from model

from helper_functions import *  # this statement imports all functions from your helper_functions file!

# Run your helper function to clear out any io files left over from old runs
# 1:
fig = go.Figure(
)

m = None

check_for_and_del_io_files()

# Make a Dash app!
app = dash.Dash(__name__)

# Define the layout.
app.layout = html.Div([
    # Section title
    html.Div([
        html.H1('Strategy'),
        html.P('This app explores a simple strategy that works as follows:'),
        html.Ol([
            html.Li([
                "While the market is not open, retrieve the past N days' " + \
                "worth of data for:",
                html.Ul([
                    html.Li("IVV, QQQ, URTH, DJ: daily open, high, low, & close prices")
                ])
            ]),
            html.Li([
                'Fit a linear trend line through the VWAP of each index/stock and record in a dataframe:',
                html.Ul([
                    html.Li('the y-intercept ("a")'),
                    html.Li('the slope ("b")')
                ]),
                '...for the fitted line.'
            ]),
            html.Li(
                'Add volatility of day-over-day log returns of IVV ' + \
                'closing prices -- observed over the past N days -- to ' + \
                'each historical data row in the FEATURES dataframe.'
            ),
            html.Li(
                'Add two RESPONSE data points to the historical FEATURES dataframe.' + \
                'The RESPONSE data includes information that communicates ' + \
                'whether when, and how a limit order to SELL IVV at a ' + \
                'price equal to (IVV Close Price of Current Trading Day) * ' + \
                '(1 + alpha) would have filled over the next n trading days.' + \
                'The second response includes information that communicates ' + \
                'whether when, and how a limit order to BUY IVV at a ' + \
                'price equal to (IVV  Close Price of Current Trading Day) * ' + \
                '(1 - alpha) would have filled over the next n trading days.'
            ),
            html.Li(
                'Using the features of a, b, of each index/stock and IVV vol alongside the ' + \
                'RESPONSE data for the past N observed trading days, ' + \
                'train a logistic regression. Use it to predict whether a ' + \
                'limit order to SELL IVV at a price equal to (IVV Close ' + \
                'Price of Current Trading Day) * (1 + alpha) would have ' + \
                'filled over the next n trading days, and whether a limit order' + \
                'to BUY IVV at a price equal to (IVV Close Price of Current Trading ' + \
                'Day) * (1 - alpha) would have filled.'
            ),
            html.Li(
                'If the regression in 6. predicts TRUE for response 1, submit two trades:'),
            html.Ul([
                html.Li(
                    'A market order to BUY lot_size shares of IVV, which ' + \
                    'fills at closing price of current trading day.'
                ),
                html.Li(
                    'A limit order to SELL lot_size shares of IVV at ' + \
                    '(current trading day close price * (1+alpha)'
                )
            ]),
            html.Li(
                'If the regression in 6. predicts TRUE for response 2, submit two trades:'),
            html.Ul([
                html.Li(
                    'A market order to SELL lot_size shares of IVV, which ' + \
                    'fills at closing price of current trading day.'
                ),
                html.Li(
                    'A limit order to BUY lot_size shares of IVV at ' + \
                    '(current trading day close price * (1-alpha)'
                )
            ]),
            html.Li(
                'If the limit orders does not fill after n days, issue a ' + \
                'market order to sell lot_size shares of IVV at close of ' + \
                'the nth day.'
            )
        ])
    ]),
    html.H1("Section 2: Modify Default Algo Parameters"),
    # Another line break
    html.Br(),
    html.Div(
        [
            "Change parameters of current model: ",
            # Your text input object goes here:
            html.Div(["Alpha: ", dcc.Input(id='alpha', value=0.01, type='number')]),
            html.Div(["N: ", dcc.Input(id="N", value=10, type='number')]),
            html.Div(["n: ", dcc.Input(id="n", value=5, type='number')]),
            html.Div(["Lot Size: ", dcc.Input(id="lot_size", value=100, type='number')]),
            html.Div(["Starting Cash: ", dcc.Input(id="start_cash", value=100000, type='number')]),
            html.Div(["Date: ", dcc.DatePickerRange(
                id='date',
                min_date_allowed=date(2002, 8, 5),
                max_date_allowed=datetime.today(),
                initial_visible_month=date(2021, 4, 1),
                start_date=date(2019, 2, 5),
                end_date=date(2021, 2, 5)
            )]),
        ],
        style={'display': 'inline-block'}
    ),
    html.Br(),
    # Submit button:
    html.Button('Backtest', id='submit-retrain', n_clicks=0),
    # Section title
    html.H1("Section 3: Backtesting Algo"),
    html.Div([
        # Candlestick graph goes here:
        dcc.Graph(id='candlestick-graph', figure=fig)
    ]),
    # Div to confirm what trade was made
    html.Div(id='output-backtest'),
    html.H1("Section 4: Model Trades"),
    html.Div(id='output-model-trades')
])


# Callback for when model is retrained
@app.callback(
    [Output('output-backtest', 'children'), Output('candlestick-graph', 'figure')],
    Input('submit-retrain', 'n_clicks'),
    [State('alpha', 'value'), State('N', 'value'), State('n', 'value'), State('lot_size', 'value'),
     State('start_cash', 'value'), State('date', 'start_date'), State('date', 'end_date')],
    # name of pair, trade amount,
    prevent_initial_call=True
)
def back_test(n_clicks, alpha, N, n, lot_size, start_cash, start_date, end_date):
    msg = ''
    print(start_date)
    print(end_date)
    m = Model(alpha=alpha, N=N, n=n, lot_size=lot_size, start_cash=start_cash, start_date=start_date, end_date=end_date)
    df = pd.read_csv('data/IVV.csv')
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )
        ]
    )
    fig.update_layout(title="IVV Candlestick Plot")
    # TODO: run back_test
    back_test_results = m.run_back_test()
    return msg, fig


# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)
