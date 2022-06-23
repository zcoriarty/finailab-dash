# Customized Bullet chart
import datetime as dt
import pandas_datareader.data as web
import plotly.express as px
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output


import datetime
import requests

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

def make_layout():

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
						update_news()
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

LINK_TABLE_HEADER = {
	'color': 'white',
	'backgroundColor': '#192734',
	'fontSize': '12px',
}

LINK_TABLE = {
	'color': 'white',
	'backgroundColor': '#192734'
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
	
# Currency pairs
currencies = ["EURUSD", "USDCHF", "USDJPY", "GBPUSD"]

# API Requests for news div
news_requests = requests.get(
    "https://newsapi.org/v2/top-headlines?sources=bbc-news&apiKey=da8e2e705b914f9f86ed2e9692e66012"
)

# API Call to update news
def update_news():
    json_data = news_requests.json()["articles"]
    df = pd.DataFrame(json_data)
    df = pd.DataFrame(df[["title", "url"]])
    max_rows = 10
    return html.Div([
			dbc.Card(
				dbc.CardBody([
					html.P(className="p-news", children="Headlines", style={'color': 'white',
						'backgroundColor': '#192734', 'fontSize':'30px', 'fontWeight':'Medium'}),
					html.P(
						className="p-news float-right",
						children="Last update : "
						+ datetime.datetime.now().strftime("%H:%M:%S"),
						style=LINK_TABLE_HEADER
					),
					html.Table(
						className="table-news",
						children=[
							html.Tr(
								children=[
									html.Td(
										children=[
											html.A(
												className="td-link",
												children=df.iloc[i]["title"],
												href=df.iloc[i]["url"],
												target="_blank"
											)
										]
									)
								]
							)
							for i in range(min(len(df), max_rows))
						], style=LINK_TABLE
					),
				]), color = '#192734'
			),
        ]
    )