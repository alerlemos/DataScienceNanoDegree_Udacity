import json
import plotly
import pandas as pd
import numpy as np
import operator
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pprint import pprint
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data//DisasterResponse.db')
#engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('Data_clean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index') 
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # caculate percentages rounded to two decimal places
    genre_per = round(100*genre_counts/genre_counts.sum(), 2)

    category_related_counts = df.groupby('related').count()['message']
    category_related_names = ['Related' if i==1 else 'Not Related' for i in list(category_related_counts.index)]

    requests_counts = df.groupby(['related','request']).count().loc[1,'message']
    category_requests_names = ['Requests' if i==1 else 'Not Requests' for i in list(requests_counts.index)]

    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values
    # Top five categories count
    top_category_count = df.iloc[:,4:].sum().sort_values(ascending=False)[1:11]
    top_category_names = list(top_category_count.index)

    # word cloud data
    #message_list = df['message'].unique().tolist()
    #messagelen_list = [len(tokenize(message)) for message in message_list]
    repeated_words=[]            # contain all repated words

    for text in df['message'].values:
        tokenized_ = tokenize(text)
        repeated_words.extend(tokenized_)

    word_count_dict = Counter(repeated_words)      # dictionary having words counts for all words\


    sorted_word_count_dict = dict(sorted(word_count_dict.items(),
                                          key=lambda item:item[1], reverse=True))
                                          # sort dictionary by\
                                                          # values
    topwords, topwords_20 =0, {}

    for k,v in sorted_word_count_dict.items():
        topwords_20[k]=v
        topwords+=1
        if topwords==20:
            break
    words=list(topwords_20.keys())
    pprint(words)
    count_props=100*np.array(list(topwords_20.values()))/df.shape[0]


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [
              {
                "type": "pie",
                #"uid": "f4de1f",
                "hole": 0.6,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": genre_per,
                  "y": genre_names
                },
                "marker": {
                  "colors": [
                    "LightSeaGreen",
                    "MediumPurple",
                    "LightSkyBlue4"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names,
                "values": genre_counts
              }
            ],
            "layout": {
              "title": "Distribution of Messages by Genre"
            }
        },

        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean,
                    #orientation = 'h',
                    marker=dict(color="MediumPurple")

                )

            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "",
                    'tickangle': -35
                    #'rotation':90
                }
            }

        },
         {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_count,
                    marker=dict(color="#17becf")
                )
            ],

            'layout': {
                'title': 'Top Ten Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_related_names,
                    y=category_related_counts,
                    marker=dict(color="LightSeaGreen")
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Related with Disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_requests_names,
                    y=requests_counts,
                    marker=dict(color="Lightsalmon")
                    #marker_color='lightsalmon'
                )
            ],

            'layout': {
                'title': 'Distribution of Request Messages <br> out of all Disaster Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
           {
            'data': [
                Bar(
                    x=words,
                    y=count_props
                )
            ],

            'layout': {
                'title': 'Frequency of top 20 words <br> as percentage',
                'yaxis': {
                    'title': 'Occurrence<br>(Out of 100)',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Top 20 words',
                    'automargin': True
                }
            }
        }
    ]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
