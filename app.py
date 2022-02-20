# Fetches and displays a basic candlestick app.

import dash
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from hw2_utils import *
from datetime import date
from model import model

# 4) Create a Dash app
app = dash.Dash(__name__)

# 5) Create the page layout
app.layout = html.Div([
    # Hidden div inside the app that stores bond features from model
    html.Div(id='model-output', style={'display': 'none'}),
    # Hidden div inside the app that stores IVV price data
    html.Div(id='ivv-historical-data', style={'display': 'none'}),
    # Date range for update historical data
    dcc.DatePickerRange(
        id='ivv-historical-data-range',
        min_date_allowed=date(2015, 1, 1),
        max_date_allowed=date.today(),
        initial_visible_month=date.today(),
        start_date=date(2021, 3, 26),
        end_date=date.today()
    ),
    html.Div(id='output-container-date-picker-range'),
    dcc.Input(id='bbg-identifier-1', type = "text", value = "IVV US Equity"),
    html.Button("UPDATE", id='update-hist-dta-button', n_clicks = 0),
    # Hidden div inside the app that stores bonds rates data
    html.Div(id='bonds-historical-data', style={'display': 'none'}),
    dcc.Graph(id='bonds-3d-graph', style={'display': 'none'})
])

@app.callback(
    [dash.dependencies.Output('bonds-historical-data', 'children'),
    dash.dependencies.Output('bonds-3d-graph', 'figure'),
    dash.dependencies.Output('bonds-3d-graph', 'style')],
    dash.dependencies.Input("update-hist-dta-button", 'n_clicks'),
    [dash.dependencies.State('ivv-historical-data-range', 'start_date'),
     dash.dependencies.State('ivv-historical-data-range', 'end_date')],
    prevent_initial_call=True
)
def update_bonds_data(n_clicks, startDate, endDate):
    # from hw2_utils import *
    # startDate = "2021-03-26"
    # endDate = "2021-03-30"

    data_years = list(
        range(pd.to_datetime(startDate).date().year,
                           pd.to_datetime(endDate).date().year + 1, 1))

    bonds_data = fetch_usdt_rates(data_years[0])

    if len(data_years) > 1:
        for year in data_years[1:]:
            bonds_data = pd.concat([bonds_data, fetch_usdt_rates(year)],
                                    axis = 0, ignore_index=True)

    bonds_data = bonds_data[bonds_data.Date >= pd.to_datetime(startDate)]
    bonds_data = bonds_data[bonds_data.Date <= pd.to_datetime(endDate)]

    fig = go.Figure(
        data=[
            go.Surface(
                z=bonds_data,
                y=bonds_data.Date,
                x=[
                    to_years(cmt_colname) for cmt_colname in list(
                        filter(lambda x: ' ' in x, bonds_data.columns.values)
                    )
                ]
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='Maturity (years)',
            yaxis_title='Date',
            zaxis_title='APR (%)',
            zaxis=dict(ticksuffix='%')
        ),
        autosize=False,
        width=1500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    return bonds_data.to_json(), fig, {'display': 'block'}

@app.callback(
    [dash.dependencies.Output('ivv-historical-data', 'children'),
    dash.dependencies.Output('output-container-date-picker-range', 'children')],
    dash.dependencies.Input("update-hist-dta-button", 'n_clicks'),
    [dash.dependencies.State("bbg-identifier-1", "value"),
    dash.dependencies.State('ivv-historical-data-range', 'start_date'),
    dash.dependencies.State('ivv-historical-data-range', 'end_date')],
    prevent_initial_call = True
)
def update_historical_data(nclicks, bbg_id_1, start_date, end_date):
    historical_data = req_historical_data(bbg_id_1, start_date, end_date)
    string_prefix = 'You have selected: '
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'Start Date: ' + start_date_string + ' | '
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'End Date: ' + end_date_string
    if len(string_prefix) == len('You have selected: '):
        string_prefix = 'Select a date to see it displayed here'
    return historical_data.to_json(), string_prefix

@app.callback(
    dash.dependencies.Output('model-output', 'children'),
    [dash.dependencies.Input('bonds-historical-data', 'children'),
    dash.dependencies.Input('ivv-historical-data', 'children')],
    prevent_initial_call = True
)
def calculate_model(bonds, ivv):
    model(bonds, ivv)

# Run it!
if __name__ == '__main__':
    app.run_server(debug=True)
