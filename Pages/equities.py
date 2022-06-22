# Customized Bullet chart
import pandas as pd
import datetime as dt
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output


# Raw Package
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

# Market Data 
import yfinance as yf
import yahoo_fin.stock_info as si

#Graphing/Visualization
import datetime as dt 
import plotly.graph_objs as go 

def make_layout(symbol):

	if symbol is None:
		symbol = 'AAPL'

	return html.Div([
		dbc.Card(
			dbc.CardBody([
				dbc.Row([
					dbc.Col([
						drawText()
					], width=2),
					dbc.Col([
						drawText()
					], width=2),
					dbc.Col([
						drawText()
					], width=2),
					dbc.Col([
						drawText()
					], width=2),
					dbc.Col([
						drawText()
					], width=2),
					dbc.Col([
						drawText()
					], width=2),
					
				]), 
				html.Br(),
				dbc.Row([
					dbc.Col([
						eps_trend(symbol),
						eps_revisions(symbol)
					], width=3),
					dbc.Col([
						centerStock(symbol)
					], width=6),
					dbc.Col([
						earnings_estimate(symbol),
						earnings_history(symbol)
					], width=3),
				], align='center'), 
				html.Br(),
				dbc.Row([
					dbc.Col([
						dcc.Tabs(id='financials-tabs', value='balance-sheet', children=[
								dcc.Tab(label='Balance Sheet', value='balance-sheet', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
								dcc.Tab(label='Income Statement', value='income-statement', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
								dcc.Tab(label='Cash Flows', value='cash-flows', style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
							], style=TABS_STYLES),
							
						
						], width=9),
				]),
				dbc.Row([
					dbc.Col([
						html.Br(),
						dcc.Loading(
							id="loading-1",
							type="default",
							children=html.Div(id="financials"),
							color = 'white'
						),
					], width=9),
					dbc.Col([
						growth_estimates(symbol)
					], width=3),
				], align='center'),      
			]), color = '#15202b' # all cell border
		)
	])



DATATABLE_STYLE = {
    'color': 'white',
    'backgroundColor': '#15202b',
}

DATATABLE_HEADER = {
	'backgroundColor': '#162636',
	'color': 'White',
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
def drawText():
	return html.Div([
		dbc.Card(
			dbc.CardBody([
				html.Div([
					html.H2("Text"),
				], style={'textAlign': 'center', 'color': 'white'}) 
			]), color = '#192734'
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

def centerStock(symbol):

	# Override Yahoo Finance 
	yf.pdr_override()

	# Create input field for our desired stock 
	stock = symbol

	# Retrieve stock data frame (df) from yfinance API at an interval of 1m 
	df = yf.download(tickers=stock,period='1d',interval='1m')

		# add Moving Averages (5day and 20day) to df 
	df['MA5'] = df['Close'].rolling(window=5).mean()
	df['MA20'] = df['Close'].rolling(window=20).mean()

	# print(df)

	# Declare plotly figure (go)
	fig=go.Figure()

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
	fig.update_layout(
		title= str(stock)+' Live Share Price:',
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

	