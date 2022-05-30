#Importing libraries
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


def upload_dataset():
    global dirnames, dirpath, filenames
    for dirpath, dirnames, filenames in os.walk(r"C:\Users\walee\Desktop\doc\Doc50"):                           
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
    for sentence in data:
        final_data.append(" ".join([st.stem(i) for i in sentence.split()]))




def prepros():
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
    feature_namess=count_vector.get_feature_names()

    tf_idf_vector=tfidf.transform(count_vector.transform(final_data))

    sorteddd_items=sort_cooo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vectorr(feature_namess,sorteddd_items,500)

    tf_idf_vector=tfidf.transform(count_vector.transform(final_data))
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    sorted_items_n = sorted_items[:500:]



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

#global variable
true_k = 5

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


gt = []

def up_gt():
    global gt
    for dpath, dnames, fnames in os.walk(r"C:\Users\walee\Downloads\WebKB\WEBKB GT"):
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

features = []

def ff_features():
    for k in keywords:
        features.append(k)

#Global
kmean_seed = []

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
        s = []
        for i in range(4):
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




def max_smax():
    global xb, xbs
    m =max(kmean_seed[0][3],kmean_seed[1][3])
    if kmean_seed[0][3] < kmean_seed[1][3]:
        sm = kmean_seed[0][3]
    else:
        sm = kmean_seed[1][3]
    for i in range(2,true_k):
        if kmean_seed[i][3] > m:
                sm = m
                m =kmean_seed[i][3]
        else:
            if kmean_seed[i][3] > sm:
                sm = kmean_seed[i][3]

    for i in range(true_k):
        if m == kmean_seed[i][3]:
            xb = i
        elif sm == kmean_seed[i][3]:
            xbs = i


#dataframes
xbest = pd.DataFrame()
xi1 = pd.DataFrame()
xj1 = pd.DataFrame()

def prepros_mutualism():
    max_smax()
    xbest = kmean_seed[xb][2].copy()
    xi1 = kmean_seed[xbs][2]
    xj1 = kmean_seed[xb][2]


#global lists
xitemp = []
xjtemp = []
tempo = []

def vec():
    for i in range(true_k):
        itemp=[]
        jtemp=[]
        j = 0
        templist_xi = []
        templist_xj = []
        countlist = []
    
    
        for index, row in xj1.iterrows():
            if row['cluster'] == i:
                templist_xj.append(row['filenames'])
    
        #print(templist_xi)
        d = len(templist_xj)
        while j<true_k:
            for index, row in xi1.iterrows():
                if row['cluster'] == j:
                    templist_xi.append(row['filenames'])
        
            #k = len(templist_xj)
            firstcount = 0
            for dd in range(0,d):
                #for kk in range(0,k):
                if templist_xj[dd] in templist_xi:
                    firstcount += 1
                

            countlist.append(firstcount)
            templist_xi.clear()
            j+=1
    
        #print(countlist)
        mmax = max(countlist)
        ib = countlist.index(mmax)
            
        if ib in tempo:
            third = first = second = -sys.maxsize
            for i in range(0, len(countlist)):
                if (countlist[i] > first):
                    third = second
                    second = first
                    first = countlist[i]
                elif (countlist[i] > second):
                    third = second
                    second = countlist[i]
                elif (countlist[i] > third):
                    third = countlist[i]
            ib = countlist.index(second)
        
        if ib in tempo:
            ib = countlist.index(third)
        
        tempo.append(ib)
    
        templist_xi.clear()    
        for index, row in xi1.iterrows():
            if row['cluster'] == ib:
                templist_xi.append(row['filenames'])
        
        #print(templist_xj)
        for file in filenames:
            if file in templist_xi:
                itemp.append(1)
            else:
                itemp.append(0)
            if file in templist_xj:
                jtemp.append(1)
            else:
                jtemp.append(0)
    
        xitemp.append(itemp)
        xjtemp.append(jtemp)



#mutualism phase begin
def f_mutual_vector():
    mutual_vector = []
    for i in range(true_k):
        tempp = []
        for j in range(len(filenames)):
            tempp.append( xitemp[i][j] * xjtemp[i][j] )
        mutual_vector.append(tempp)
    return mutual_vector



import random
def rand_mutual_v(m_v):
    for i in range(true_k):
        for j in range(30):
            while True:
                index = random.randint(0, 929)
                if m_v[i][index] == 1:
                    continue
                else:
                    m_v[i][index] = 1
                    for k in range(true_k):
                        if k != i:
                            m_v[k][index] = 0
                    break



def fill_mutual_vector(mmvv):
    for i in range(true_k):
        for j in range(len(filenames)):
            if (xjtemp[i][j] == 1 and mmvv[i][j] == 1) or (xjtemp[i][j] == 1 and mmvv[i][j] == 0):
                mmvv[i][j]=1
            else:
                mmvv[i][j]=0


xinew = pd.DataFrame()
check_df = pd.DataFrame()
def begin_mutualism_ph1():
    pppp=0
    pcheck = pppp
    generations = 0
    while generations < 10000:
        mutual_vector = f_mutual_vector()
        fill_mutual_vector(mutual_vector)
        rand_mutual_v(mutual_vector)
        xinew = pd.DataFrame(columns = ['filenames','cluster'])
        for i in range(5):
            for j in range(len(filenames)):
                if mutual_vector[i][j] == 1:
                    xinew = xinew.append({"filenames": filenames[j], "cluster": i}, ignore_index=True)
                
        xinew = xinew.sort_values(by=['cluster'])
        xinew.reset_index(drop = True, inplace = True)

        pp_info  = purity_calculation(xinew,true_k,gt)
        pppp = c_purity(pp_info)
        
        if pppp > pcheck:
            pcheck = pppp
            check_df = pd.DataFrame(columns = ['filenames','cluster'])
            check_df = xinew.copy()
            
        generations+=1

    xinew = pd.DataFrame(columns = ['filenames','cluster'])
    xinew = check_df.copy()


xjnew = pd.DataFrame()
def begin_mutualism_ph2():
    pppp=0
    global pcheck
    pcheck = pppp
    generations = 0
    while generations < 10000:
        mutual_vector = f_mutual_vector()
        fill_mutual_vector(mutual_vector)
        rand_mutual_v(mutual_vector)
        xjnew = pd.DataFrame(columns = ['filenames','cluster'])
        for i in range(true_k):
            for j in range(len(filenames)):
                if mutual_vector[i][j] == 1:
                    xjnew = xjnew.append({"filenames": filenames[j], "cluster": i}, ignore_index=True)
                
        xjnew = xjnew.sort_values(by=['cluster'])
        xjnew.reset_index(drop = True, inplace = True)

        pp_info  = purity_calculation(xjnew,true_k,gt)
        pppp = c_purity(pp_info)
        
        if pppp > pcheck:
            pcheck = pppp
            check_df = pd.DataFrame(columns = ['filenames','cluster'])
            check_df = xjnew.copy()
            print(pcheck)
            
        generations+=1
    xjnew = pd.DataFrame(columns = ['filenames','cluster'])
    xjnew = check_df.copy()




def min_smin():
    global xmin, xmins
    smin =max(kmean_seed[0][3],kmean_seed[1][3])
    if kmean_seed[0][3] < kmean_seed[1][3]:
        min_ = kmean_seed[0][3]
    else:
        min_ = kmean_seed[1][3]
    for i in range(2,5):
        if kmean_seed[i][3] < min_:
                smin = min_
                min_ =kmean_seed[i][3]
        else:
            if kmean_seed[i][3] < smin:
                smin = kmean_seed[i][3]

    for i in range(5):
        if min_ == kmean_seed[i][3]:
            xmin = i
        elif smin == kmean_seed[i][3]:
            xmins = i


def post_pros():
    min_smin()
    pp_info  = purity_calculation(xinew,5,gt)
    ppi = c_purity(pp_info)


    kmean_seed[xmin][2] = xinew
    kmean_seed[xmin][3] = ppi
    kmean_seed[xmins][2] = xjnew
    kmean_seed[xmins][3] = pcheck



def prepros_commensalism():
    max_smax()
    xbest = kmean_seed[xb][2].copy()
    xi1 = kmean_seed[xbs][2]
    xj1 = kmean_seed[xb][2]


    xitemp.clear()
    xjtemp.clear()
    tempo.clear()
    vec()



def fcomm_vec():
    comm_vec = []
    for i in range(true_k):
        temp = []
        for j in range(len(filenames)):
            if xjtemp[i][j] == 1:
                temp.append(1)
            else:
                temp.append(0)
        comm_vec.append(temp)
    return comm_vec



def commensalism(cvec):
    for i in range(true_k):
        for j in range(len(filenames)):
            if (xitemp[i][j] == 1 and xjtemp[i][j] == 0):
                cvec[i][j]=1
                for k in range(true_k):
                    if k!=i:
                        cvec[k][j] = 0



def rand_commensalism(cvec):
    for i in range(true_k):
        for j in range(40):
            while True:
                index = random.randint(0, (len(filenames)-1))
                if cvec[i][index] == 1:
                    continue
                else:
                    cvec[i][index] = 1
                    for k in range(true_k):
                        if k != i:
                            cvec[k][index] = 0
                    break


def begin_commensalism():
    pppp=0
    pcheck = pppp
    generations = 0;
    while generations < 10000:
        test_vector = fcomm_vec()
        commensalism(test_vector)
        rand_commensalism(test_vector)
        xinew = pd.DataFrame(columns = ['filenames','cluster'])
        for i in range(true_k):
            for j in range(len(filenames)):
                if test_vector[i][j] == 1:
                    xinew = xinew.append({"filenames": filenames[j], "cluster": i}, ignore_index=True)
                
        xinew = xinew.sort_values(by=['cluster'])
        xinew.reset_index(drop = True, inplace = True)

        pp_info  = purity_calculation(xinew,true_k,gt)
        pppp = c_purity(pp_info)
        
        if pppp > pcheck:
            pcheck = pppp
            check_df = pd.DataFrame(columns = ['filenames','cluster'])
            check_df = xinew.copy()
            
        generations+=1
    xinew = pd.DataFrame(columns = ['filenames','cluster'])
    xinew = check_df.copy()



def postpros_commensalism():
    min_smin()
    kmean_seed[xmin][2] = xinew
    kmean_seed[xmin][3] = pcheck




def prepros_commensalism():
    max_smax()
    xbest = kmean_seed[xb][2].copy()
    xi1 = kmean_seed[xbs][2]
    xj1 = kmean_seed[xb][2]
    xitemp.clear()
    xjtemp.clear()
    tempo.clear()
    vec()




def fparasite_vec_i():
    para_vec = []
    for i in range(true_k):
        temp = []
        for j in range(len(filenames)):
            if xitemp[i][j] == 1:
                temp.append(1)
            else:
                temp.append(0)
        para_vec.append(temp)
    return para_vec

def fparasite_vec_j():
    para_vec = []
    for i in range(true_k):
        temp = []
        for j in range(len(filenames)):
            if xjtemp[i][j] == 1:
                temp.append(1)
            else:
                temp.append(0)
        para_vec.append(temp)
    return para_vec




def parasitism(pvec_i,pvec_j):
    for i in range(true_k):
        for j in range(len(filenames)):
            if (xitemp[i][j] == 1 and xjtemp[i][j] == 0):
                pvec_j[i][j]=1
                pvec_i[i][j]=0
                while True:
                    indexi = random.randint(0, (true_k-1))
                    if indexi == i:
                        continue
                    else:
                        pvec_i[indexi][j]=1
                        break
                for k in range(true_k):
                    if k!=i:
                        pvec_j[k][j] = 0


def rand_parasitism(pvec):
    for i in range(true_k):
        for j in range(30):
            while True:
                index = random.randint(0, (len(filenames)-1))
                if pvec[i][index] == 1:
                    continue
                else:
                    pvec[i][index] = 1
                    for k in range(true_k):
                        if k != i:
                            pvec[k][index] = 0
                    break


def begin_parasitism_ph1():
    pppp=0
    pcheck = pppp
    generations = 0;
    while generations < 10000:
        test_vector1 = fparasite_vec_i()
        test_vector2 = fparasite_vec_j()
        parasitism(test_vector1,test_vector2)
        rand_parasitism(test_vector2)
        xjnew = pd.DataFrame(columns = ['filenames','cluster'])
        for i in range(true_k):
            for j in range(len(filenames)):
                if test_vector2[i][j] == 1:
                    xjnew = xjnew.append({"filenames": filenames[j], "cluster": i}, ignore_index=True)
                
        xjnew = xjnew.sort_values(by=['cluster'])
        xjnew.reset_index(drop = True, inplace = True)

        pp_info  = purity_calculation(xjnew,true_k,gt)
        
        pppp = c_purity(pp_info)
        
        if pppp > pcheck:
            pcheck = pppp
            check_df = pd.DataFrame(columns = ['filenames','cluster'])
            check_df = xjnew.copy()
            print(pcheck)
            
        generations+=1
    xjnew = pd.DataFrame(columns = ['filenames','cluster'])
    xjnew = check_df.copy()




def postpros_parasitism():
    min_smin()            
    kmean_seed[xmin][2] = xjnew
    kmean_seed[xmin][3] = pcheck



if __name__ == "__main__":
    upload_dataset()
    ds_path()
    stemming()
    print(final_data[0])
