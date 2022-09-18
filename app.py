# import dash libraries
import dash
from dash import DiskcacheManager, CeleryManager, html, dcc, dash_table
import plotly.express as px
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.dependencies import Input, Output, State
from datetime import timedelta, datetime, date

import diskcache

# hosting on heroku
import gunicorn
from whitenoise import WhiteNoise

# import finance libraries
import yfinance as yf
yf.pdr_override()

# import DS libraries
import pandas as pd
import numpy as np
import json

# import files
from Pages import equities as eq
from Pages import crypto, fx, equity_visuals, code_modal

import multiprocessing, threading




# get tickers
sp_tickers = pd.read_csv('Static/Data/sp500_companies.csv', usecols=['Symbol'])
sp_tickers = sp_tickers['Symbol'].values.tolist()


crypto_tickers = pd.read_csv('Static/Data/crypto_tickers.csv', names=['Symbol'])
crypto_tickers = crypto_tickers['Symbol'].values.tolist()

fx_countries = pd.read_csv('Static/Data/Foreign_Exchange_Rates.csv')
fx_countries = fx_countries.replace('ND', np.nan) 
fx_countries = fx_countries.dropna()

country_lst = list(fx_countries.columns[2:])

equity_df = pd.DataFrame()

# get all companies from json file
with open("Static/Dropdown Data/companies.json", "r") as read_file:
	company_list = json.load(read_file)
company_options_list = []
for company in company_list:
	# company_options_list.append({'label': str(company_list[company] + ' (' + company + ')'),
	# 							'value': company})
    company_options_list.append(company)

tickers_dict = {'/': company_list, '/equities': company_list, '/equity-visuals': [], '/crypto': crypto_tickers, '/FX': country_lst, '/fixed-income': [], '/commodities': [], '/sentiment': [], }
names = list(tickers_dict.keys())
nested_options = tickers_dict[names[0]]

# NEED TO BE CHANGED TO CELERY FOR PRODUCTION

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

server = app.server

server.wsgi_app = WhiteNoise(server.wsgi_app, root='Static/') 

eq.register_callbacks(app)
equity_visuals.register_callbacks(app)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 54,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#15202b",
	"color": "#FFFFFF",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 54,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#15202b",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "16rem",
    "margin-bottom": "30rem",
    "margin-right": 0,
    "padding": "2rem 1rem",
    "background-color": "#15202b",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    # "margin-left": "2rem",
    # "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#15202b",
}
SEARCH_STYLE  = {
    "background-color": "#15202b",
    'color': 'black',
    }
DATATABLE_STYLE = {
    'color': 'white',
    'backgroundColor': '#15202b',
}

# with open('Static/Markdown Code/equity_mkdn.md', 'r') as text:
#     code = text.read() 

# Sticky dash board header
navbar = dbc.NavbarSimple(
    children=[
        dbc.Button("See Code", id="open-modal", className="me-1", outline=True, color="primary", n_clicks=0),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Python Code")),
                dbc.ModalBody(dcc.Markdown(id="see-code-content")),
            ],
            id="modal-content",
            size="lg",
            is_open=False,
            centered=True
        ),
        dbc.Button("Sidebar", outline=True, color="secondary", className="mr-1", id="btn_sidebar"),
        
    ],
    sticky="top",
    brand="FinAILab Dash",
    brand_href="#",
    color="#15202b",
    dark=True,
    fluid=True,
)

# sidebar, including input for content window
sidebar = html.Div(
    [
        html.Br(),
		
        dbc.Nav(
            [
                html.Br(),
                dbc.DropdownMenu(label="Equities", children = [dbc.DropdownMenuItem("Visualizations", href="/equity-visuals", id="equity-visuals-link"), 
                                                                dbc.DropdownMenuItem("Data", href="/equities", id="equities-link"),
                                                                ]
                                , menu_variant="dark", nav=True, group=True
                ),
                dbc.NavLink("Crypto", href="/crypto", id="crypto-link"),
                dbc.NavLink("FX", href="/FX", id="FX-link"),
                dbc.NavLink("Fixed Income", href="/fixed-income", id="fixed-income-link"),
                dbc.NavLink("Commodities", href="/commodities", id="commodities-link"),
                dbc.NavLink("Sentiment", href="/sentiment", id="sentiment-link"),
                html.Br(),
                dcc.Dropdown(id="selected-symbol", style=SEARCH_STYLE, clearable=False, placeholder='Select Ticker...'),
                html.Br(),
                # dcc.DatePickerRange(
                #     id='my-date-picker-range',
                #     min_date_allowed=date(1995, 8, 5),
                #     max_date_allowed=date.today(),
                #     initial_visible_month=date.today(),
                #     end_date=date.today()
                # ),
                # dmc.DatePicker(
                #     id='start-date',
                #     label="Start Date",
                #     inputFormat="DD-MM-YYYY",
                #     minDate=datetime(1995, 8, 5),
                #     maxDate=datetime.now()- timedelta(1),
                #     value=datetime.now() - timedelta(1),
                #     style=SEARCH_STYLE,
                #     clearable=False
                # ),
                # dmc.DatePicker(
                #     id='end-date',
                #     label="End Date",
                #     inputFormat="DD-MM-YYYY",
                #     minDate=datetime(1995, 8, 5),
                #     maxDate=datetime.now(),
                #     value=datetime.now(),
                #     style=SEARCH_STYLE,
                #     clearable=False
                # ),
                
            ],
            vertical=True,
            pills=True,
            
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)


content = html.Div(
	style=CONTENT_STYLE, 
	id="page-content",
	)


app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ]
)



# toggle see code button in dash header
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

app.callback(
    Output("modal-content", "is_open"),
    Input("open-modal", "n_clicks"),
    State("modal-content", "is_open"),
)(toggle_modal)

categories = ['equities','equity-visuals', 'crypto', 'FX', 'fixed-income', 'commodities', 'sentiment']
# adjust dropdown tickers for a given tab
@app.callback(Output('selected-symbol', 'options'), [Input("url", "pathname")]
)
def update_dropdown(name):

    
    if name == 'FX':
        return [{'label': i, 'value': i} for i in tickers_dict[name]]
    if name in ['equities','equity-visuals']:
        return [{'label': str(company_list[company] + ' (' + company + ')'), 'value': company} for company in tickers_dict[name]]
    else:
        return [{'label': i, 'value': i} for i in tickers_dict[name]]

@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"{i}-link", "active") for i in categories],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False, False, False, False, False
    return [pathname == f"/{i}" for i in categories]

# communicate with "see code" content dictionary
@app.callback(Output("see-code-content", "children"), [Input("url", "pathname"), Input('selected-symbol', 'value')])
def render_code(pathname, symbol):
    return code_modal.get_modal_content(pathname, symbol)
    
@app.callback(Output("page-content", "children"), [Input("url", "pathname"), Input('selected-symbol', 'value')])
def render_page_content(pathname, symbol):
    if pathname in ["/", "/equities"]:
        return eq.make_layout(symbol)
    elif pathname in ["/equity-visuals"]:
        return equity_visuals.make_layout()
    elif pathname == "/crypto":
        return crypto.make_layout(symbol)
    elif pathname == "/FX":
        return fx.make_layout(symbol)
    elif pathname == "/fixed-income":
        return html.P("Oh cool, this is page 3!")
    elif pathname == "/commodities":
        return html.P("Oh cool, this is page 4!")
    elif pathname == "/sentiment":
        return html.P("Oh cool, this is page 5!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        dbc.Container(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ],
            fluid=True,
            className="py-3",
        ),
        className="p-3 bg-light rounded-3",
    )

# # @app.callback(Output("nothing", "children"), Input("url", "pathname"), background=True, manager=background_callback_manager)
# def get_yf_data():
    
#     for section in range(50, len(company_options_list), 50):
#         # Retrieve stock data frame (df) from yfinance API at an interval of 1m 
#         yf_pd = yf.download(tickers=company_options_list,period='1d',interval='1m', group_by='ticker', auto_adjust = False, prepost = False, threads = True, proxy = None, progress=False)
#         # equity_df.append(df)
#         print("yf", yf_pd)
#         eq.yf_data = pd.concat([yf_pd, eq.yf_data])
#     return html.P(id='placeholder')

    

if __name__ == "__main__":

    app.run_server(debug=True, port=8086)

    
    
    
    