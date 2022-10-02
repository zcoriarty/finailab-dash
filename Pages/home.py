import warnings
warnings.filterwarnings('ignore')
import os
import tempfile
import zipfile

# Customized Bullet chart
import datetime as dt
# import pandas_datareader.data as web
import plotly.express as px
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import plotly.express as px
from plotly.tools import mpl_to_plotly
import dash.dependencies
import pyfolio as pf
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import empyrical
import quantstats as qs
from quantstats import stats
from pandas_datareader import data as web
from plotly.subplots import make_subplots


# Raw Package
import numpy as np
import pandas as pd
# from pandas_datareader import data as pdr

# Market Data 
import yfinance as yf
import yahoo_fin.stock_info as si

#Graphing/Visualization
import plotly.graph_objs as go

from Pages import fx



# global yf_data
# yf_data = pd.DataFrame()
df_dict = {}

fx_countries = pd.read_csv('Static/Data/Foreign_Exchange_Rates.csv')
fx_countries = fx_countries.replace('ND', np.nan) 
fx_countries = fx_countries.dropna()

def make_layout(symbol, start, end):

	if symbol in fx_countries:
		return fx.make_layout(symbol)

	if symbol is None:
		symbol = 'AAPL'
		# app.equity_df.append(yf.download(tickers='AAPL',period='1d',interval='1m', group_by='ticker', auto_adjust = False, prepost = False, threads = True, proxy = None))

	return html.Div([
		dbc.Card(
			dbc.CardBody([
				html.Br(),
				dbc.Row([
					dbc.Col([
						centerStock(symbol, start, end)
					], width=12),
				], align='center'), 
				html.Br(),
	
			]), color = PRIMARY, style ={'border-radius': 10} # all cell border
		)
	], style={'margin-bottom':'30rem'})
		


PRIMARY = '#FFFFFF' 
SECONDARY = '#FFFFFF'
ACCENT = '#98C1D9'
DARK_ACCENT = '#474747'
SIDEBAR = '#F7F7F7'

# PRIMARY = '#15202b'
# SECONDARY = '#192734'
# ACCENT = '#FFFFFF'
# SIDEBAR = '#F4511E'
#F4511E

DATATABLE_STYLE = {
    'color': 'white',
    'backgroundColor': PRIMARY,
}

DATATABLE_HEADER = {
	'backgroundColor': SIDEBAR,
	'color': 'white',
	'fontWeight': 'bold',
}

TABS_STYLES = {
    'height': '44px'
}
TAB_STYLE = {
    'padding': '15px',
    'fontWeight': 'bold',
	'color': DARK_ACCENT,
	'backgroundColor': SECONDARY,
	'borderRadius': '10px',
	'margin-left': '6px',
}

TAB_SELECTED_STYLE = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': ACCENT,
    'color': PRIMARY,
    'padding': '15px',
	'borderRadius': '10px',
	'margin-left': '6px',
}

# helper function for closing temporary files
def close_tmp_file(tf):
    try:
        os.unlink(tf.name)
        tf.close()
    except:
        pass

# # add csv to download folder
# def add_csv_to_folder(df, name):
# 	filepath = Path('/finailab_dash/Static/download_folder/' + name + '.csv')
# 	filepath.parent.mkdir(parents=True, exist_ok=True)
# 	df.to_csv(filepath)

def beautify_plotly(fig):
	return html.Div([
			dbc.Card(
				dbc.CardBody([
					dcc.Graph(
						figure=fig,
					config={
						'displayModeBar': False
					}
					)
				]), color = SECONDARY, style ={'border-radius': 10}
			),  
		])


# create main equity plot
def centerStock(symbol, start, end):

	from plotly.subplots import make_subplots

	# Override Yahoo Finance 
	yf.pdr_override()

	delta = dt.datetime.strptime(end, '%Y-%m-%d') - dt.datetime.strptime(start, '%Y-%m-%d')
	if delta.days < 30:
		# Retrieve stock data frame (df) from yfinance API at an interval of 1m 
		df = yf.download(tickers=symbol,period='1d',interval='1m', start=start,end=end)
	else:
		df = yf.download(tickers=symbol,period='1d',start=start,end=end)
	
	df.drop(df.tail(1).index,inplace=True)
	# add_csv_to_folder(df, "center_stock")
	df_dict['Download Data'] = df
	# print(yf_data)
	# df = pd.DataFrame(yf_data[symbol])

	# add Moving Averages (5day and 20day) to df 
	df['MA5'] = df['Close'].rolling(window=5).mean()
	df['MA20'] = df['Close'].rolling(window=20).mean()

	# print(df)

	# Declare plotly figure (go)
	fig=go.Figure()

	# Creating figure with second y-axis
	fig = make_subplots(specs=[[{'secondary_y': True}]])

	# Adding line plot with close prices and bar plot with trading volume
	fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=symbol+' Close'), secondary_y=False)
	fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.5, marker_color=['black'], marker_colorscale='Rainbow',), secondary_y=True)

	fig.add_trace(go.Candlestick(x=df.index,
					open=df['Open'],
					high=df['High'],
					low=df['Low'],
					close=df['Close'], name = 'market data'))

	# Add 5-day Moving Average Trace
	fig.add_trace(go.Scatter(x=df.index, 
							y=df['MA5'], 
							opacity=0.7, 
							line=dict(color='blue', width=2), 
							name='MA 5'))
	# Add 20-day Moving Average Trace
	fig.add_trace(go.Scatter(x=df.index, 
							y=df['MA20'], 
							opacity=0.7, 
							line=dict(color='orange', width=2), 
							name='MA 20'))

	fig.update_xaxes(
		# rangeslider_visible=True,
		rangeselector=dict(
			buttons=list([
				dict(count=15, label='15m', step='minute', stepmode='backward'),
				dict(count=45, label='45m', step='minute', stepmode='backward'),
				dict(count=1, label='HTD', step='hour', stepmode='todate'),
				dict(count=3, label='3h', step='hour', stepmode='backward'),
				dict(step='all')
			]), bgcolor = SECONDARY
		),
		nticks=delta.days * 4
	)
	fig.layout.xaxis.type = 'category'
	

	# Updating layout
	fig.update_layout(
		xaxis_rangeslider_visible=True,
		hovermode='x'
	)

	fig.update_layout(
		title= str(symbol)+' Live Share Price:',
		yaxis_title='Stock Price (USD per Shares)',
		# template='plotly_dark',
		plot_bgcolor= SECONDARY,
		paper_bgcolor= SECONDARY,   
		font=dict(color=DARK_ACCENT),
	)



	return html.Div([
			dbc.Card(
				dbc.CardBody([
					dcc.Graph(
						figure=fig,
					config={
						'displayModeBar': False
					}
					)
				]), color = SECONDARY, style ={'border-radius': 10}
			),  
		])

def register_callbacks(app):
	@app.callback(
		Output("download-center-stock-csv", "data"),
		[Input("center_stock", "n_clicks"), Input("center_stock", "children")],
		prevent_initial_call=True,
	)
	def func(n_clicks, name):
		df = df_dict[name]
		return dcc.send_data_frame(df.to_csv, "finailab_data.csv")




	# @app.callback(Output('financials', 'children'), Input('financials-tabs', 'value'), Input('selected-symbol', 'value')
	# )
	# def render_financials(tab, symbol):

	# 	if symbol is None:
	# 		symbol = 'AAPL'

	# 	if tab == 'balance-sheet':
	# 		return balance_sheet(symbol)
	# 	elif tab == 'income-statement':
	# 		return income_statement(symbol)
	# 	elif tab == 'cash-flows':
	# 		return cash_flows(symbol)

	