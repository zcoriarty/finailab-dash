import warnings

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

# global yf_data
# yf_data = pd.DataFrame()

def make_layout(symbol):

	if symbol is None:
		symbol = 'AAPL'
		# app.equity_df.append(yf.download(tickers='AAPL',period='1d',interval='1m', group_by='ticker', auto_adjust = False, prepost = False, threads = True, proxy = None))

	sharpe_ratio, max_drawdown, cumulative_returns_plot, annual_monthly_returns_plot, rolling_sharpe_plot, drawdown_periods_plot, drawdown_underwater_plot = key_metrics(symbol)


	return html.Div([
		dbc.Card(
			dbc.CardBody([
				# dbc.Row([
				# 	dbc.Col([
				# 		drawText("Sharpe Ratio", sharpe_ratio)
				# 	], width=4),
				# 	dbc.Col([
				# 		drawText("Max Drawdown", max_drawdown)
				# 	], width=4),
				# 	# dbc.Col([
				# 	# 	drawText()
				# 	# ], width=4),
				# # 	dbc.Col([
				# # 		drawText()
				# # 	], width=2),
				# # 	dbc.Col([
				# # 		drawText()
				# # 	], width=2),
				# # 	dbc.Col([
				# # 		drawText()
				# # 	], width=2),
					
				# ]), 
				html.Br(),
				dbc.Row([
					# dbc.Col([
					# 	eps_trend(symbol),
					# 	eps_revisions(symbol)
					# ], width=3),
					dbc.Col([
						centerStock(symbol)
					], width=9),
					dbc.Col([
						drawText("Sharpe Ratio", sharpe_ratio),
						html.Br(),
						drawText("Max Drawdown", max_drawdown)
					], width=3),
				], align='center'), 
				html.Br(),
				dbc.Tabs(
					[
						dbc.Tab(cumulative_returns_plot, label="Cumulative Returns"),
						dbc.Tab(annual_monthly_returns_plot, label="Annual and Monthly Returns"),
						dbc.Tab(rolling_sharpe_plot, label="Rolling Sharpe"),
						dbc.Tab(drawdown_periods_plot, label="unfinished"),
						dbc.Tab(drawdown_underwater_plot, label="Drawdown Underwater"),
						# dbc.Tab(quantiles_plot, label="Scatter"),
					],
					id="tabs",
					# active_tab="tab-1",
				),
				# dbc.Row([
				# 	dbc.Col([
				# 		dcc.Tabs(id='financials-tabs', value='balance-sheet', children=[
				# 				dcc.Tab(label='Balance Sheet', value='balance-sheet', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
				# 				dcc.Tab(label='Income Statement', value='income-statement', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
				# 				dcc.Tab(label='Cash Flows', value='cash-flows', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
				# 			], style=TABS_STYLES),
							
						
				# 		], width=9),
				# ]),
				# dbc.Row([
				# 	dbc.Col([
				# 		html.Br(),
				# 		dcc.Loading(
				# 			id="loading-1",
				# 			type="default",
				# 			children=html.Div(id="financials"),
				# 			color = 'white'
				# 		),
				# 	], width=9),
				# 	dbc.Col([
				# 		growth_estimates(symbol)
				# 	], width=3),
				# ], align='center'),      
			]), color = '#15202b' # all cell border
		)
	], style={'margin-bottom':'30rem'})



DATATABLE_STYLE = {
    'color': 'white',
    'backgroundColor': '#15202b',
}

DATATABLE_HEADER = {
	'backgroundColor': '#162636',
	'color': 'white',
	'fontWeight': 'bold',
}

TABS_STYLES = {
    'height': '44px'
}
TAB_STYLE = {
    'padding': '15px',
    'fontWeight': 'bold',
	'color': 'white',
	'backgroundColor': '#192734',
	'borderRadius': '10px',
	"margin-left": "6px",
}

TAB_SELECTED_STYLE = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'white',
    'color': '#15202b',
    'padding': '15px',
	'borderRadius': '10px',
	"margin-left": "6px",
}

# Text field
def drawText(title, text):
	return html.Div([
		dbc.Card(
			dbc.CardBody([
				html.Div([
					dbc.CardHeader(title),
					html.Br(),
					html.H3(text),
				], style={'textAlign': 'center', 'color': 'white'})
			]), color = '#192734', style={"height": 225},
		),
	])

# def topBar():
# 	# Data
# 	end = dt.datetime.now()
# 	start = end - dt.timedelta(hours = 12)

# 	stocks = web.DataReader(['^GSPC', '^DJI', '^IXIC'], 'yahoo', start, end)
# 	stocks_close = pd.DataFrame(web.DataReader(['^GSPC', '^DJI', '^IXIC'], 'yahoo', start, end)['Close'])


# 	c_bullet = go.Figure()

# 	c_bullet.add_trace(go.Indicator(
# 		mode = "number+gauge+delta", 
# 		value = int(stocks_close['^GSPC'].tail(1)),
# 		delta = {'reference': int(stocks_close['^GSPC'].tail(2)[0])},
# 		domain = {'x': [0.25, 1], 
# 				'y': [0.08, 0.25]},
# 		# title = {'text':"<b>S&P DAY<br>RANGE</b><br><span style='color: gray; font-size:0.8em'>U.S. $</span>", 
# 		# 		'font': {"size": 5}},    
# 		gauge = {
# 			'shape': "bullet",
# 			'axis': {'range': [None, 550]},
# 			'threshold': {
# 				'line': {'color': "Red", 'width': 2},
# 				'thickness': 0.75,
# 				'value': 505},
# 			'steps': [
# 				{'range': [0, 350], 'color': "gray"},
# 				{'range': [350, 550], 'color': "lightgray"}],
# 			'bar': {'color': 'black'}}))

# 	# c_bullet.add_trace(go.Indicator(
# 	# 	mode = "number+gauge+delta", 
# 	# 	value = int(stocks_close['^DJI'].tail(1)),
# 	# 	delta = {'reference': int(stocks_close['^DJI'].tail(2)[0])},
# 	# 	domain = {'x': [0.25, 1], 
# 	# 			'y': [0.4, 0.6]},
# 	# 	title = {'text':"<b>DJI DAY<br>RANGE</b><br><span style='color: gray; font-size:0.8em'>U.S. $</span>", 
# 	# 			'font': {"size": 14}},
# 	# 	gauge = {
# 	# 		'shape': "bullet",
# 	# 		'axis': {'range': [None, 1800]},
# 	# 		'threshold': {
# 	# 			'line': {'color': "red", 'width': 2},
# 	# 			'thickness': 0.75,
# 	# 			'value': 1681},
# 	# 		'steps': [
# 	# 			{'range': [0, 1300], 'color': "gray"},
# 	# 			{'range': [1300, 1800], 'color': "lightgray"}],
# 	# 		'bar': {'color': 'black'}}))

# 	# c_bullet.add_trace(go.Indicator(
# 	# 	mode = "number+gauge+delta", 
# 	# 	value = int(stocks_close['^IXIC'].tail(1)),
# 	# 	delta = {'reference': int(stocks_close['^IXIC'].tail(2)[0])},
# 	# 	domain = {'x': [0.25, 1], 
# 	# 			'y': [0.7, 0.9]},
# 	# 	title = {'text':"<b>NASDAQ DAY<br>RANGE</b><br><span style='color: gray; font-size:0.8em'>U.S. $</span>", 
# 	# 			'font': {"size": 14}},
# 	# 	gauge = {
# 	# 		'shape': "bullet",
# 	# 		'axis': {'range': [None, 250]},
# 	# 		'threshold': {
# 	# 			'line': {'color': "red", 'width': 2},
# 	# 			'thickness': 0.75,
# 	# 			'value': 208},
# 	# 		'steps': [
# 	# 			{'range': [0, 150], 'color': "gray"},
# 	# 			{'range': [150, 250], 'color': "lightgray"}],
# 	# 		'bar': {'color': "black"}}))

# 	c_bullet.update_layout(height = 18, margin = {'t':0, 'b':0, 'l':0})
# 	return html.Div([
# 		dbc.Card(
# 			dbc.CardBody([
# 				dcc.Graph(figure=c_bullet)
# 				])
# 		)
# 	])
# def background_processing():

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
				]), color = '#192734'
			),  
		])

def centerStock(symbol):

	from plotly.subplots import make_subplots

	# Override Yahoo Finance 
	yf.pdr_override()

	# Retrieve stock data frame (df) from yfinance API at an interval of 1m 
	df = yf.download(tickers=symbol,period='1d',interval='1m')
	# print(yf_data)
	# df = pd.DataFrame(yf_data[symbol])

	# add Moving Averages (5day and 20day) to df 
	df['MA5'] = df['Close'].rolling(window=5).mean()
	df['MA20'] = df['Close'].rolling(window=20).mean()

	# print(df)

	# Declare plotly figure (go)
	fig=go.Figure()

	# Creating figure with second y-axis
	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# Adding line plot with close prices and bar plot with trading volume
	fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=symbol+' Close'), secondary_y=False)
	fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.5, marker_color=['black'], marker_colorscale="Rainbow",), secondary_y=True)

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
				dict(count=15, label="15m", step="minute", stepmode="backward"),
				dict(count=45, label="45m", step="minute", stepmode="backward"),
				dict(count=1, label="HTD", step="hour", stepmode="todate"),
				dict(count=3, label="3h", step="hour", stepmode="backward"),
				dict(step="all")
			]), bgcolor = '#192734'
		)
	)
	

	# Updating layout
	fig.update_layout(
		xaxis_rangeslider_visible=True,
		hovermode='x'
	)

	fig.update_layout(
		title= str(symbol)+' Live Share Price:',
		yaxis_title='Stock Price (USD per Shares)',
		template='plotly_dark',
		plot_bgcolor= '#192734',
		paper_bgcolor= '#192734',   
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
				]), color = '#192734'
			),  
		])

def key_metrics(symbol):
	
	def get_max_drawdown_underwater_f(underwater):
		"""
		Determines peak, valley, and recovery dates given an 'underwater'
		DataFrame.

		An underwater DataFrame is a DataFrame that has precomputed
		rolling drawdown.

		Parameters
		----------
		underwater : pd.Series
		Underwater returns (rolling drawdown) of a strategy.

		Returns
		-------
		peak : datetime
			The maximum drawdown's peak.
		valley : datetime
			The maximum drawdown's valley.
		recovery : datetime
			The maximum drawdown's recovery.
		"""

		#valley = np.argmin(underwater)  # end of the period
		valley = underwater.index[np.argmin(underwater)] # end of the period

		# Find first 0
		peak = underwater[:valley][underwater[:valley] == 0].index[-1]
		# Find last 0
		try:
			recovery = underwater[valley:][underwater[valley:] == 0].index[0]
		except IndexError:
			recovery = np.nan  # drawdown not recovered
		return peak, valley, recovery

	def get_symbol_returns_from_yahoo_f(symbol, start=None, end=None):
		"""
		Wrapper for pandas.io.data.get_data_yahoo().
		Retrieves prices for symbol from yahoo and computes returns
		based on adjusted closing prices.

		Parameters
		----------
		symbol : str
			Symbol name to load, e.g. 'SPY'
		start : pandas.Timestamp compatible, optional
			Start date of time period to retrieve
		end : pandas.Timestamp compatible, optional
			End date of time period to retrieve

		Returns
		-------
		pandas.DataFrame
			Returns of symbol in requested period.
		"""

		try:
			px = web.get_data_yahoo(symbol, start=start, end=end)
			px['date'] = px.index.to_list()
			#px['date'] = px['date'].apply(lambda x: pd.Timestamp(x))
			#px['date'] = pd.to_datetime(px['date'])
			#px['date'] = pd.to_datetime(px['date'], unit='s')
			px.set_index('date', drop=False, inplace=True)
			
			#px.index.rename('date',inplace=True)
			rets = px[['Adj Close']].pct_change().dropna()
			rets.rename(columns={"Adj Close": "adjclose"},inplace=True)
		except Exception as e:
			warnings.warn(
				'Yahoo Finance read failed: {}, falling back to Google'.format(e),
				UserWarning)
			px = web.get_data_google(symbol, start=start, end=end)
			rets = px[['Close']].pct_change().dropna()

		rets.index = rets.index.tz_localize("UTC")
		rets.columns = [symbol]
		return rets

	

	empyrical.utils.get_symbol_returns_from_yahoo = get_symbol_returns_from_yahoo_f
	pf.timeseries.get_max_drawdown_underwater = get_max_drawdown_underwater_f
	# return pf.create_returns_tear_sheet(stock_rets)

	stock_rets = pf.utils.get_symbol_rets(symbol)

	sharpe_ratio = empyrical.sharpe_ratio(stock_rets)
	max_drawdown = empyrical.max_drawdown(stock_rets)

	def cumulative_returns_plot():
		
		# extract data from pyfolio func
		plt = pf.plotting.plot_returns(stock_rets)
		xy_data = plt.get_lines()[0].get_data()
		
		# create plotly fig
		df = pd.DataFrame(xy_data).T
		fig = px.line(df, x=0, y=1)

		fig.update_layout(
			title= 'Rolling Sharpe Ratio',
			yaxis_title='Returns',
			xaxis_title='Date',
			template='plotly_dark',
			plot_bgcolor= '#192734',
			paper_bgcolor= '#192734',
		)

		return beautify_plotly(fig)
		
	
	def annual_monthly_returns_plot():
		fig = make_subplots(rows=1, cols=3)
		df = pd.DataFrame(stock_rets)
		df['month'] = pd.DatetimeIndex(df.index).month
		df['year'] = pd.DatetimeIndex(df.index).year
		df[symbol] = df[symbol] * 100

		fig1 = px.histogram(df, x=symbol)

		fig2 = px.bar(df, x=symbol, y="year", orientation='h')

		fig3 = go.Figure(data=go.Heatmap(
				z=df[symbol],
				x=df['month'],
				y=df['year'],
				colorscale='YlGn'))

		fig = make_subplots(rows=1, cols=3)

		for d in fig1.data:
			fig.add_trace((go.Scatter(x=d['x'], y=d['y'], name = d['name'])), row=1, col=1)
				
		for d in fig2.data:
			fig.add_trace((go.Bar(x=d['x'], y=d['y'],  name = d['name'], orientation='h')), row=1, col=2)

		for d in fig3.data:
			fig.add_trace((go.Heatmap(z=df[symbol], x=df['month'], y=df['year'], colorscale='YlGn')), row=1, col=3)

		fig.update_layout(
			template='plotly_dark',
			plot_bgcolor= '#192734',
			paper_bgcolor= '#192734',
		)

		return beautify_plotly(fig)
	
	# def quantiles_plot():
	# 	fig = pf.plot_return_quantiles(stock_rets)
	# 	return (fig)
	
	def rolling_sharpe_plot():
		fig = pf.plot_rolling_sharpe(stock_rets)
		xy_data = fig.get_lines()[0].get_data()
				
		# create plotly fig
		df = pd.DataFrame(xy_data).T
		fig = px.line(df, x=0, y=1)

		fig.update_layout(
			title= 'Rolling Sharpe Ratio',
			yaxis_title='Sharpe Ratio',
			xaxis_title='Year',
			template='plotly_dark',
			plot_bgcolor= '#192734',
			paper_bgcolor= '#192734',
		)

		return beautify_plotly(fig)

	# NOT FINISHED
	def drawdown_periods_plot():
		fig = pf.plot_drawdown_periods(stock_rets)
		xy_data = fig.get_lines()[0].get_data()
				
		# create plotly fig
		df = pd.DataFrame(xy_data).T
		fig = px.line(df, x=0, y=1)

		fig.update_layout(
			title= 'Top 10 Drawdown Periods',
			yaxis_title='Cumulative Returns',
			xaxis_title='Year',
			template='plotly_dark',
			plot_bgcolor= '#192734',
			paper_bgcolor= '#192734',
		)

		return beautify_plotly(fig)
	
	def drawdown_underwater_plot():
		fig = pf.plot_drawdown_underwater(stock_rets)
		xy_data = fig.get_lines()[0].get_data()
				
		# create plotly fig
		df = pd.DataFrame(xy_data).T
		fig = px.area(df, x=0, y=1)

		fig.update_layout(
			title= 'Underwater Plot',
			yaxis_title='Drawdown',
			xaxis_title='Year',
			template='plotly_dark',
			plot_bgcolor= '#192734',
			paper_bgcolor= '#192734',
		)

		return beautify_plotly(fig)

	return sharpe_ratio, max_drawdown, cumulative_returns_plot(), annual_monthly_returns_plot(), rolling_sharpe_plot(), drawdown_periods_plot(), drawdown_underwater_plot()




def balance_sheet(symbol):

	ticker = symbol
	data = yf.Ticker(ticker)

	df = pd.DataFrame(data.balance_sheet).T

	return html.Div([
			dbc.Card(
				dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
								style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
				]), color = '#192734'
			),  
		])

def eps_trend(symbol):

	ticker = symbol
	df = si.get_analysts_info(ticker)['EPS Trend'].assign(hack='').set_index('hack')
	return html.Div([
		dbc.Card(
			dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
							style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
			]), color = '#192734'
		),  
	])

def growth_estimates(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['Growth Estimates'].assign(hack='').set_index('hack')
	return html.Div([
		dbc.Card(
			dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
							style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
			]), color = '#192734'
		),  
	])

def earnings_estimate(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['Earnings Estimate'].assign(hack='').set_index('hack')
	return html.Div([
		dbc.Card(
			dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
							style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
			]), color = '#192734'
		),  
	])
def revenue_estimate(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['Revenue Estimate'].assign(hack='').set_index('hack')
	return html.Div([
		dbc.Card(
			dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
							style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
			]), color = '#192734'
		),  
	])
def earnings_history(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['Earnings History'].assign(hack='').set_index('hack')
	return html.Div([
		dbc.Card(
			dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
							style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
			]), color = '#192734'
		),  
	])
def eps_revisions(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['EPS Revisions'].assign(hack='').set_index('hack')
	return html.Div([
		dbc.Card(
			dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
							style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
			]), color = '#192734'
		),  
	])

def income_statement(symbol):

	ticker = symbol
	data = yf.Ticker(ticker)

	df = pd.DataFrame(data.financials).T
	return html.Div([
			dbc.Card(
				dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
								style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
				]), color = '#192734'
			),  
		])

def cash_flows(symbol):

	ticker = symbol
	data = yf.Ticker(ticker)

	df = pd.DataFrame(data.cashflow).T
	return html.Div([
			dbc.Card(
				dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
								style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER,style_table={'overflowX': 'auto'})
				]), color = '#192734'
			),  
		])


def register_callbacks(app):

	@app.callback(Output('financials', 'children'), Input('financials-tabs', 'value'), Input('selected-symbol', 'value')
	)
	def render_financials(tab, symbol):

		if symbol is None:
			symbol = 'AAPL'

		if tab == 'balance-sheet':
			return balance_sheet(symbol)
		elif tab == 'income-statement':
			return income_statement(symbol)
		elif tab == 'cash-flows':
			return cash_flows(symbol)

	