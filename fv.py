import pandas as pd
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
import re
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def payload(timestamp, start, end):
    t = int(timestamp)
    return 1 if start <= t and t <= end else 0

def main():
    features = ['time1', 'id1', 'ppid1', 'pid1', 'exe1', 'time2', 'id2', 'ppid2', 'pid2', 'exe2', 'time3', 'id3', 'ppid3', 'pid3', 'exe3']
    try:
        fmatrix = pd.read_csv('feature_matrix.csv', header=0)
    except:
        fmatrix = pd.DataFrame({})
        for ftype in ['benign', 'malicious']:
            for i in range(1,50):
                fvfile = 'fv_' + ftype + str(i) + '.out'
                timefile = 'times_' + ftype + str(i) + '.txt'
                fv = pd.read_csv(fvfile, header=0)
                with open(timefile) as t:
                    times = re.findall('(start|end)=(\d+)', t.read())
                    start = int(times[0][1])
                    end = int(times[1][1])
                if ftype == 'benign':
                    fv['label'] = [0]*len(fv)
                else:
                    fv['label'] = fv['time1'].apply(payload, args=(start, end))
                fmatrix = pd.concat((fmatrix, fv), ignore_index=True)

        for f in features:
            fmatrix[f] = LE().fit_transform(fmatrix[f])
        fmatrix.to_csv('feature_matrix.csv', index=False)

    #clf = xgb.XGBClassifier()
    clf = RandomForestClassifier()
    le = {f: LE() for f in features}
    train, test = train_test_split(fmatrix)
    clf.fit(train[features], train['label'])
    ypred = clf.predict_proba(test[features])
    fpr, tpr, _ = roc_curve(test['label'], ypred[:,1])

    layout = Layout(                                                                                     
           title='ROC curve for random forest with all features',                                               
           xaxis=dict(title='False positive rate'),                                                        
           yaxis=dict(title='True positive rate'),                                                         
           showlegend=True                                                                                 
    )                                                                                                        
    fig = Figure(data = [Scatter(x=fpr, y=tpr, mode='lines', name='AUC %f' % auc(fpr,tpr))], layout=layout)
    py.plot(fig, filename='rf feature roc', auto_open=False)                                          

if __name__ == '__main__':
    main()
