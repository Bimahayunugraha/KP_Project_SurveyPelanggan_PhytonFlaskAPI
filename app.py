import flask
from flask import jsonify, json
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import plotting
import plotly.offline as py
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')


app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app, support_credentials=True)

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@app.route('/api/resources/surveypelanggan/data', methods=['GET'])
def api_data():
    data = pd.read_csv('input/Survey_Kepuasan_Pelanggan_2.csv')

    # Pelayanan Pelanggan 
    size = data['Pelayanan Pelanggan'].value_counts().tolist()

    # Kepuasan Pelanggan
    size2 = data['Kepuasan Pelanggan'].value_counts().tolist()

    # Customer Service
    size3 = data['Customer Service'].value_counts().tolist()

    # Metode Pembayaran
    size4 = data['Metode Pembayaran'].value_counts().tolist()

    # Kualitas Internet
    size5 = data['Kualitas Internet'].value_counts().tolist()
    
    return jsonify(size, size2, size3, size4, size5)

@app.route('/api/resources/surveypelanggan/kmeansdata', methods=['GET'])
def api_data2():
    data = pd.read_csv('input/Survey_Kepuasan_Pelanggan_2.csv')
    x = data.iloc[:, [4, 5]].values
 

    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, random_state = 0)
        km.fit(x)
        wcss.append(km.inertia_)
    
    return jsonify(wcss)

app.run()

@app.route('/api/resources/surveypelanggan/clusteringdata', methods=['GET'])
def api_data3():
    data = pd.read_csv('input/Survey_Kepuasan_Pelanggan_2.csv')
    x = data.iloc[:, [4, 5]].values
 
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters = 7, init = 'k-means++', max_iter = 300, n_init = 5, random_state = 0)
    ymeans = km.fit_predict(x).tolist()
    
    return jsonify(ymeans)

app.run()