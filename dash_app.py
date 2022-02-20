from statistics import stdev

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from dash_table import DataTable, FormatTemplate
from dash.dependencies import Input, Output, State
import pandas as pd
import gc
import pickle
from datetime import date, datetime
from time import sleep
from momentum import *  # import everything from model

from helper_functions import *  # this statement imports all functions from your helper_functions file!

# Run your helper function to clear out any io files left over from old runs
# 1:
fig = go.Figure(
)

check_for_and_del_io_files()

# Make a Dash app!
app = dash.Dash(__name__)

# Define the layout.
app.layout = html.Div([
    # Section title
    html.Div([
        html.H1("Section 1: Model Description"),
        html.H2('Strategy'),
        html.P('This app explores a simple strategy that works as follows:'),
        html.Ol([
            html.Li([
                "While the market is not open, retrieve the past N days' " + \
                "worth of data for:",
                html.Ul([
                    html.Li("IVV, QQQ, URTH, DJ: daily open, high, low, & close prices")
                ])
            ]),
            html.Br(),
            html.Li([
                'Fit a linear trend line through the VWAP of each index/stock and record in a dataframe:',
                html.Ul([
                    html.Li('the y-intercept ("a")'),
                    html.Li('the slope ("b")')
                ]),
                '...for the fitted line.'
            ]),
            html.Br(),
            html.Li(
                'Add volatility of day-over-day log returns of IVV ' + \
                'closing prices -- observed over the past N days -- to ' + \
                'each historical data row in the FEATURES dataframe.', style={'width': '70%'}
            ),
            html.Br(),
            html.Li(
                'Add two RESPONSE data points to the historical FEATURES dataframe.' + \
                'The RESPONSE data includes information that communicates ' + \
                'whether when, and how a limit order to SELL IVV at a ' + \
                'price equal to (IVV Close Price of Current Trading Day) * ' + \
                '(1 + alpha) would have filled over the next n trading days.' + \
                'The second response includes information that communicates ' + \
                'whether when, and how a limit order to BUY IVV at a ' + \
                'price equal to (IVV  Close Price of Current Trading Day) * ' + \
                '(1 - alpha) would have filled over the next n trading days.', style={'width': '70%'}
            ),
            html.Br(),
            html.Li(
                'Using the features of a, b, of each index/stock and IVV vol alongside the ' + \
                'RESPONSE data for the past N observed trading days, ' + \
                'train a logistic regression. Use it to predict whether a ' + \
                'limit order to SELL IVV at a price equal to (IVV Close ' + \
                'Price of Current Trading Day) * (1 + alpha) would have ' + \
                'filled over the next n trading days, and whether a limit order' + \
                'to BUY IVV at a price equal to (IVV Close Price of Current Trading ' + \
                'Day) * (1 - alpha) would have filled.', style={'width': '70%'}
            ),
            html.Br(),
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
            html.Br(),
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
            html.Br(),
            html.Li(
                'If the limit orders does not fill after n days, close the order on the nth day.'
            )
        ]),
        html.H2('Parameters'),
        html.Ol([
            html.Li(["Alpha: profit seeking percentage (i.e. 5%)"]),
            html.Br(),
            html.Li(["N (time to close): time before market closes (i.e. 5)"]),
            html.Br(),
            html.Li(["n: window size (i.e. 3 days)"]),
            html.Br(),
            html.Li(["Lot Size: total number of shares in one trade (i.e. 50)"]),
            html.Br(),
            html.Li(["Starting crash: Initial value of total portfolio "]),
            html.Br(),
            html.Li(["Date: The date array for the trading algorithm"]),
        ])
    ]),
    html.H1("Section 2: Modify Default Algo Parameters"),
    # Another line break
    html.Div(
        [
            "Change parameters of current model: ",
            # Your text input object goes here:
            html.Div(["Alpha: ", dcc.Input(id='alpha', value=0.03, type='number')],
                     style={'display': 'table-cell', 'padding': 3, 'verticalAlign': 'middle'}),
            html.Div(["N: ", dcc.Input(id="N", value=10, type='number')],
                     style={'display': 'table-cell', 'padding': 3, 'verticalAlign': 'middle'}),
            html.Div(["n: ", dcc.Input(id="n", value=5, type='number')],
                     style={'display': 'table-cell', 'padding': 3, 'verticalAlign': 'middle'}),
            html.Div(["Lot Size: ", dcc.Input(id="lot_size", value=100, type='number')],
                     style={'display': 'table-cell', 'padding': 3, 'verticalAlign': 'middle'}),
            html.Div(["Starting Cash: ", dcc.Input(id="start_cash", value=100000, type='number')],
                     style={'display': 'table-cell', 'padding': 3, 'verticalAlign': 'middle'}),
            html.Div(["Model Backtest Dates: ", dcc.DatePickerRange(
                id='date',
                min_date_allowed=date(2002, 8, 5),
                max_date_allowed=datetime.today(),
                initial_visible_month=date(2021, 4, 1),
                start_date=date(2018, 2, 5),
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
    html.Div(id='backtest-output'),
    # Div to confirm what trade was made
    html.Div(
        [dcc.Graph(id='alpha-beta')],
        style={'display': 'inline-block', 'width': '50%'}
    ),
    html.Table(
        [html.Tr([
            html.Th('Alpha'), html.Th('Beta'),
            html.Th('Geometric Mean Return'),
            html.Th('Average Trades per Year'),
            html.Th('Volatility'), html.Th('Sharpe')
        ])] + [html.Tr([
            html.Td(html.Div(id='strategy-alpha')),
            html.Td(html.Div(id='strategy-beta')),
            html.Td(html.Div(id='strategy-gmrr')),
            html.Td(html.Div(id='strategy-trades-per-yr')),
            html.Td(html.Div(id='strategy-vol')),
            html.Td(html.Div(id='strategy-sharpe'))
        ])],
        className='main-summary-table',
        style={'display': 'inline-block', 'width': '50%'}
    ),
    html.Div([
        html.H2(
            'Trade Ledger',
            style={
                'display': 'inline-block', 'text-align': 'center',
                'width': '100%'
            }
        ),
        DataTable(
            id='trade-ledger',
            fixed_rows={'headers': True},
            style_cell={'textAlign': 'center'},
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
    ]),
    html.Div([
        html.Div([
            html.H2(
                'Calendar Ledger',
                style={
                    'display': 'inline-block', 'width': '45%',
                    'text-align': 'center'
                }
            ),
            html.H2(
                'Trade Blotter',
                style={
                    'display': 'inline-block', 'width': '55%',
                    'text-align': 'center'
                }
            )
        ]),
        html.Div(
            DataTable(
                id='calendar-ledger',
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'center'},
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
            style={'display': 'inline-block', 'width': '45%'}
        ),
        html.Div(
            DataTable(
                id='blotter',
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'center'},
                style_table={'height': '300px', 'overflowY': 'auto'}
            ),
            style={'display': 'inline-block', 'width': '55%'}
        )
    ])
])


# Callback for when model is retrained
@app.callback(
    [Output('blotter', 'data'),
     Output('blotter', 'columns'),
     Output('calendar-ledger', 'data'),
     Output('calendar-ledger', 'columns'),
     Output('trade-ledger', 'data'),
     Output('trade-ledger', 'columns'),
     Output('candlestick-graph', 'figure'),
     Output('alpha-beta', 'figure'),
     Output('strategy-alpha', 'children'),
     Output('strategy-beta', 'children'),
     Output('strategy-gmrr', 'children'),
     Output('strategy-trades-per-yr', 'children'),
     Output('strategy-vol', 'children'),
     Output('strategy-sharpe', 'children'),
     Output('backtest-output', 'children')
     ],
    Input('submit-retrain', 'n_clicks'),
    [State('alpha', 'value'), State('N', 'value'), State('n', 'value'), State('lot_size', 'value'),
     State('start_cash', 'value'), State('date', 'start_date'), State('date', 'end_date')],
    # name of pair, trade amount,
    prevent_initial_call=True
)
def back_test(n_clicks, alpha, N, n, lot_size, start_cash, start_date, end_date):
    gc.collect()
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
    ledger, blotter, portfolio, final_msg = m.run_back_test()
    trade_ledger = ledger.to_dict('records')
    trade_ledger_columns = [
        dict(id="ID", name="Trade ID"),
        dict(id="Date Created", name="Date Created"),
        dict(id="Date Closed", name="Date Closed"),
        dict(id="Position Type", name="Position Type"),
        dict(id="Entry Price", name="Entry Price", type='numeric',
             format=FormatTemplate.money(2)),
        dict(id="Exit Price", name="Exit Price", type='numeric',
             format=FormatTemplate.money(2)),
        dict(id="Benchmark Entry", name="Benchmark Entry", type='numeric',
             format=FormatTemplate.money(2)),
        dict(id="Benchmark Exit", name="Benchmark Exit", type='numeric',
             format=FormatTemplate.money(2)),
        dict(id="Return on Trade", name="Return on Trade", type='numeric',
             format=FormatTemplate.percentage(3)),
        dict(id="Benchmark Return", name="Benchmark Return", type='numeric',
             format=FormatTemplate.percentage(3))
    ]
    trade_blotter = blotter.to_dict("records")
    trade_blotter_columns = [
        dict(id="ID", name="Trade ID"),
        dict(id="Date Created", name="Date Created"),
        dict(id="Action", name="Action"),
        dict(id="Size", name="Size"),
        dict(id="Symbol", name="Symbol"),
        dict(id="Order Price", name="Order Price", type='numeric',
             format=FormatTemplate.money(2)),
        dict(id="Type", name="Type"),
        dict(id="Status", name="Status"),
        dict(id="Fill Price", name="Fill Price", type='numeric',
             format=FormatTemplate.money(2)),
        dict(id="Fill/Cancelled Date", name="Fill/Cancelled Date")
    ]
    trade_portfolio = portfolio.to_dict("records")
    trade_portfolio_columns = [
        dict(id="Date", name="Date"),
        dict(id="Cash", name="Available Cash", type='numeric',
             format=FormatTemplate.money(2)),
        dict(id="Num Shares", name="Number of Shares"),
        dict(id="Share total Value", name="Total Value of Shares", type='numeric',
             format=FormatTemplate.money(2)),
        dict(id="Total Port Value", name="Total Value", type='numeric',
             format=FormatTemplate.money(2))
    ]
    clean_ledger = ledger.dropna()
    X = clean_ledger["Benchmark Return"].values.reshape(-1, 1)
    lin_reg = LinearRegression().fit(X, clean_ledger["Return on Trade"])
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = lin_reg.predict(x_range.reshape(-1, 1))
    fig2 = px.scatter(
        clean_ledger,
        title="Performance against Benchmark",
        x='Benchmark Return',
        y='Return on Trade',
        color="Position Type"
    )
    fig2.add_traces(go.Scatter(x=x_range, y=y_range, name='OLS Fit'))
    alpha = str(round(lin_reg.intercept_ * 100, 3)) + "% / trade"
    beta = round(lin_reg.coef_[0], 3)
    gmrr = (clean_ledger['Return on Trade'] + 1).product() ** (
            1 / len(clean_ledger)) - 1
    bench_gmr = (clean_ledger['Benchmark Return'] + 1).product() ** (
            1 / len(clean_ledger)) - 1
    avg_trades_per_yr = round(
        clean_ledger['Date Created'].groupby(
            pd.DatetimeIndex(clean_ledger['Date Created']).year
        ).agg('count').mean(),
        0
    )
    vol = stdev(clean_ledger['Return on Trade'])
    sharpe = round(gmrr / vol, 3)
    gmrr_str = str(round(gmrr, 3))
    vol_str = str(round(vol, 3))
    bal = start_cash
    for i in clean_ledger['Benchmark Return']:
        bal += lot_size * (1+i)
    final_msg += f" with GMMR {round(gmrr*100, 2)}%, Benchmark final balance ${round(bal, 2)} with GMMR {round(bench_gmr*100, 2)}%"
    return trade_blotter, trade_blotter_columns, trade_portfolio, trade_portfolio_columns, trade_ledger, trade_ledger_columns, fig, fig2, alpha, beta, gmrr_str, avg_trades_per_yr, vol_str, sharpe, final_msg


# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)
