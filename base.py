from flask import Flask, render_template, request
from ast import keyword
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sys
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

#Global variables
filenames = []
dirpath =[]
dirnames=[]
data = []
final_data = []
count_vector = []
count_matrix = []
tfidf = []
tfidf_matrix = [] 
keywords = []
gt = []
features = []
kmean_seed = []


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method =='POST':
        final_data.clear()
        filepath = request.form['fp']
        upload_dataset(filepath)
        ds_path()
        stemming()
    return render_template("index.html", final_data = final_data)

@app.route('/kmeans', methods=['GET', 'POST'])
def kmeanpage():
    global true_k
    if request.method == 'POST':
        gt.clear()
        true_k = int(request.form['kvalue'])
        gtpath = request.form['fp']
        if len(gtpath) != 0:
            up_gt(gtpath)
            prepros()
            top_fea()
            ff_features()
            kmeanseed()
    return render_template("kmeans.html" , gt=gt)

@app.route('/kmeanresults', methods=['POST', 'GET'])
def kmeanresults():
    return render_template('kmeanresults.html',  kmean_seed=kmean_seed)

@app.route('/sos', methods= ['GET', 'POST'])
def sospage():
    if request.method == 'POST':
        print("hello")
    return render_template("sos.html")

def upload_dataset(file_path):
    global dirnames, dirpath, filenames
    for dirpath, dirnames, filenames in os.walk(file_path):                           
        #print(filenames)
        continue

def ds_path():
    global data
    i = 0
    for file in filenames:
        tempdata= ""
        fpath = os.getcwd()
        fpath = os.path.join(fpath, str(dirpath))
        fpath = os.path.join(fpath, str(filenames[i]))                              
        filee = open(fpath,mode='r')
        tempdata = filee.read()                                                      
        tempdata = str.lower(tempdata)
        tempdata = ''.join(e for e in tempdata if e.isalnum() or e==' ')            
        tempdata = re.sub(' +', ' ',tempdata)                                        
        data.append(tempdata)                                                 
        filee.close()
        i=i+1


def stemming():
    st = PorterStemmer()

    global final_data
    final_data.clear()
    for sentence in data:
        final_data.append(" ".join([st.stem(i) for i in sentence.split()]))

def prepros():
    global count_vector, count_matrix, tfidf , tfidf_matrix
    count_vector = CountVectorizer(stop_words="english", max_df=0.85)
    count_matrix = count_vector.fit_transform(final_data)


    tfidf = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf.fit(count_matrix)

    tfidf_matrix = tfidf.transform(count_matrix)

def sort_cooo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vectorr(feature_namess, sorteddd_items, topn):
    """get the feature names and tf-idf score of top n items"""
    
    sorteddd_items = sorteddd_items[:topn]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorteddd_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_namess[idx])

    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col,coo_matrix.row, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[2],x[1], x[0]), reverse=True)    

def top_fea():
    global keywords, sorted_items_n
    feature_namess=count_vector.get_feature_names()

    tf_idf_vector=tfidf.transform(count_vector.transform(final_data))

    sorteddd_items=sort_cooo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vectorr(feature_namess,sorteddd_items,100)

    tf_idf_vector=tfidf.transform(count_vector.transform(final_data))
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    sorted_items_n = sorted_items[:100:]

def mElbow():
    Sum_of_squared_distances = []
    K = range(2,10)

    for k in K:
        km = KMeans(n_clusters=k, max_iter=200, n_init=10)
        km = km.fit(count_matrix)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def kmeans(tfidf_matrx):
    model = KMeans(n_clusters=true_k,  init='k-means++', max_iter=300, n_init=20)
    model = model.fit(tfidf_matrx)
    return model.labels_

def to_dataframe(file_n, label):
    x = pd.DataFrame(list(zip(file_n,label)),columns=['filenames','cluster'])
    #x = x.sort_values(by=['cluster'])
    x = x.sort_values(by=['cluster'])
    x.reset_index(drop = True, inplace = True)
    return x

def up_gt(gtpath):
    global gt
    for dpath, dnames, fnames in os.walk(gtpath):
        gt.append(fnames)
    del gt[0]

def purity_calculation(clusterinf,tk,gt):
    templist = []
    tempo = []
    i = 0
    j = len(gt)
    count = [0]*tk
    while i < tk:
        for index, row in clusterinf.iterrows():
                if row['cluster'] == i:
                    templist.append(row['filenames'])
        d = len(templist)
        it = 0
        first_i = 0
        firstcount = [0]*j
        for jj in range(j):
            k = len(gt[it])
            for kk in range(k):
                for dd in range(d):
                    if gt[jj][kk] == templist[dd]:
                        firstcount[first_i] += 1
                        break
            first_i+=1
            it+=1
        
        mmax = max(firstcount)
        ib = firstcount.index(mmax)
            
        if ib in tempo:
            max_=max(firstcount[0],firstcount[1])
            if firstcount[0] < firstcount[1]:
                secondmax = firstcount[0]
            else:
                secondmax = firstcount[1]
            for n in range(2,len(firstcount)):
                if firstcount[n]>max_:
                    secondmax=max_
                    max_=firstcount[n]
                else:
                    if firstcount[n]>secondmax:
                        secondmax=firstcount[n]
            ib = firstcount.index(secondmax)
        
        tempo.append(ib)
        
        count[i] = firstcount[ib]
        #it+=1
        i+=1
        templist.clear()

    return count

def c_purity(p_info):    
    c = 0
    Total = len(filenames)
    for i in range(0, len(p_info)):
        c += p_info[i]
    purity = (c/Total)*100
    return purity

def ff_features():
    global features
    for k in keywords:
        features.append(k)

def kmeanseed():
    global kmean_seed
    for i in range(true_k):
        s = []
        labels = kmeans(tfidf_matrix)
        cluster_info = to_dataframe(filenames, labels)
        pur = purity_calculation(cluster_info, true_k, gt)
        purity = c_purity(pur)
        
        labels.sort()
        #d_dict = dict()
        #for index, row in cluster_info.iterrows():
        #    d_dict.update({row['filenames'] : row['cluster']})
        s.append(features)
        s.append(labels)
        s.append(cluster_info)
        s.append(purity)
            
        kmean_seed.append(s)

def seedfile_save():
    with open("kmeanseeds(webkb).txt",'w') as f:
        for i in range(5):
            f.write(f"SEED #{i+1}\n")
            for j in range(4):
                f.write(str(kmean_seed[i][j]))
                f.write("\n")    


if __name__ == "__main__":
    app.run(debug=True)