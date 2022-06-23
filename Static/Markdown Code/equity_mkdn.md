
```python
# Market Data
import yfinance as yf
from pandas_datareader import data as pdr
import yahoo_fin.stock_info as si

# Data manipulation
import numpy as np
import pandas as pd

# Visualizations
import plotly.graph_objs as go 
```

### Center Equity Graph:
```python

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

	return fig
```
### Various Data frames

```python
def balance_sheet(symbol):

	ticker = symbol
	data = yf.Ticker(ticker)

	df = pd.DataFrame(data.balance_sheet).T

def eps_trend(symbol):

	ticker = symbol
	df = si.get_analysts_info(ticker)['EPS Trend'].assign(hack='').set_index('hack')

def growth_estimates(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['Growth Estimates'].assign(hack='').set_index('hack')
	

def earnings_estimate(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['Earnings Estimate'].assign(hack='').set_index('hack')
	
def revenue_estimate(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['Revenue Estimate'].assign(hack='').set_index('hack')
	
def earnings_history(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['Earnings History'].assign(hack='').set_index('hack')
	
def eps_revisions(symbol):
	ticker = symbol
	df = si.get_analysts_info(ticker)['EPS Revisions'].assign(hack='').set_index('hack')
	

def income_statement(symbol):

	ticker = symbol
	data = yf.Ticker(ticker)

	df = pd.DataFrame(data.financials).T

def cash_flows(symbol):

	ticker = symbol
	data = yf.Ticker(ticker)

	df = pd.DataFrame(data.cashflow).T
	
```
