# -*- coding: utf-8 -*-

deploy=False

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os

import pandas as pd
import classification_utils as utils


if deploy:
    from flask import Flask
    server = Flask(__name__)
    server.secret_key = os.environ.get('secret_key', 'secret')
    app = dash.Dash(__name__, server=server)
else:        
    app = dash.Dash('Cloud classification training')

app.title = 'CUMULO Training Tracker | Dash'

classification_file = ('classify_128km-classifications.csv')
data = utils.read_and_parse_classifications(classification_file, annotations_parser=utils.tidy_parser)
users = list(set(data.user_name.values))

all_label_counts = data.groupby('label').count().annotations

overview_figure ={
                'data': [
                    go.Bar(
                        x=all_label_counts.index.values,
                        y=all_label_counts.values,
                        text=['{:.1%} of 1000 sample goal'.format(i) for i in all_label_counts.values/1000],
                    )
                ],
                'layout': go.Layout(
                    xaxis={'tickangle': 45},
                    yaxis={'title': '# of classifications'},
                    margin={'l': 40, 'b': 160, 't': 10, 'r': 100},
                    legend={'x': 0, 'y': 1},
#                     hovermode='closest'
                )
            }

text_style = dict(color='#444', fontFamily='sans-serif', fontWeight=300)
head_style = text_style.copy(); head_style.update({'color': '#FFFFFF'})




app.layout = html.Div([
        html.Div([
            html.H1('Cloud Classification Training Set Tracker', style=head_style),
        ], style={'width': '90%', 'height': '100px', 'background-image': 'url(/assets/MODIS_img.jpg)', 'background-size': '100%', 'margin': 'auto', 'text-align': 'center'}),
        
        html.Div([
            html.H3('Summary Statistics', style=text_style),
            html.P('Total progress on generation classification training data', style=text_style),
            dcc.Graph(id='summary_graph', figure=overview_figure),
        ], style={'width':  '49%', 'vertical-align': 'top', 'display': 'inline-block'}),
    
    
        html.Div([
            html.H3('User Statistics', style=text_style),
            dcc.Dropdown(id='user_select', placeholder='Select a user',
                         options=[{'label': x, 'value': x} for x in users]),
#             html.P(id='user_header', style=text_style),
            dcc.Graph(id='user_count_graph'),
        ], style={'width':  '49%', 'vertical-align': 'top', 'display': 'inline-block'}),
    ])


# @app.callback(Output('user_header', 'children'), [Input('user_select', 'value')])
# def user_select_callback(text_input):
#     return 'Showing classification results for <b>{}</b>.'.format(text_input)

@app.callback(Output('user_count_graph', 'figure'), [Input('user_select', 'value')])
def user_graph_callback(text_input):
    user_counts = data[data.user_name == text_input].groupby('label').count().annotations
    figure={
                'data': [
                    go.Bar(
                        x=user_counts.index.values,
                        y=user_counts.values/sum(user_counts.values),
#                         text=df[df['continent'] == i]['country'],
#                         mode='markers',
#                         opacity=0.7,
#                         marker={
#                             'size': 15,
#                             'line': {'width': 0.5, 'color': 'white'}
#                         },
#                         name=i
                    ) #for i in df.continent.unique()
                ],
                'layout': go.Layout(
#                     xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                    yaxis={'title': 'frac. of classifications'},
                    margin={'l': 40, 'b': 160, 't': 10, 'r': 100},
                    legend={'x': 0, 'y': 1},
#                     hovermode='closest'
                )
            }
    return figure

if __name__ == '__main__':
    
#     if deploy:
#         pass
# #         app.run_server()
#     else:

    app.run_server(debug=True)

