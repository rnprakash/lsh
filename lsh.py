#! /usr/bin/python

import sys
import numpy as np
import random as rand
from collections import defaultdict
from sets import Set
from sklearn.metrics import roc_curve, auc

def deriv(ts):
    return np.ediff1d(np.array(ts))

class Hash:
    def feature_map(ts):
        return np.concatenate((ts, np.ediff1d(np.array(ts))))

    def distance(x,y):
        return np.linalg.norm(x-y)

    # \phi is feature mapping, d is distance function
    def __init__(self, phi=feature_map, d=distance):
        self.phi = phi
        self.d = d
        self.A = defaultdict(dict)
        self.planes = defaultdict(dict)

    def delte_model(self):
        self.A.clear()

    def train(self, t, R, K = [3], a = 5, b = 3,  M = 2**16):
        self.R = R
        self.K = K
        self.a = a
        self.M = M
        self.b = b

        for i in range(len(t)):
            for k in self.K:
                if ((i * 2) % k) != 0:
                    continue
                tk = t[-k:]
                D = self.hash(tk)
                for d,r in zip(D,self.R):
                    for b,v in enumerate(d):
                        try:
                            self.A[r][b].add(v)
                            #self.A[r][0].add(d[0])
                            #self.A[r][1].add(d[1])
                        except:
                            self.A[r][b] = Set([v])
                            #self.A[r][0] = Set([d[0]])
                            #self.A[r][1] = Set([d[1]])
        return sum(len(self.A[k]) for k in self.A)

    # t is a real-time time series
    # i is index to current time
    # behaviors is true if known for k, false if anomalous, None otherwise
    def behavior_hash(self, t, i):
        behaviors = []
        for k in self.K:
            if ((i * 2) % k) != 0:
                continue
            tk = t[-k:]
            # D, indexed by r \in R, is distance per units in R
            D = self.hash(tk)
            # Check for anomaly
            if self.is_anomalous(D):
                behaviors.append((i,k))
        return behaviors

    def is_anomalous(self, D):
        # A[r] is dictionary of known values for threshold r. 0 is known, 1 is not
        # known is 2D list of behavior values
        anomalous = [ 0 if ( all( d[b] in self.A[r][b] for b in range(self.b) ) ) else 1 for d,r in zip(D,self.R) ]
        thresholds = [ sum( anomalous[i:] ) for i in range(len(anomalous)/2) ]
        return any( thresholds[i] > i/2 for i in range(len(thresholds)) )

    # Hash time series with window length w
    # Window length is a variable parameter that defines the sensitivity of the hash
    def hash(self, tk):
        # Compute feature map on tk
        p = self.phi(tk)
        # Project p onto $a$ random hyperplanes
        try:
            P = [ [self.project(p, plane) for plane in self.planes[len(p)][b]] for b in range(self.b) ]
        except Exception as e:
            for b in range(self.b):
                self.planes[len(p)][b] = self.make_planes(len(p))
            P = [ [self.project(p, plane) for plane in self.planes[len(p)][b] ] for b in range(self.b) ]
        # Get minimum distance from origin of points per unit r
        distances = [ np.array([self.d(np.zeros(len(p)), p) for p in P[b]]) for b in range(self.b) ]
        D = [ [ int(min( d/r )) for d in distances ] for r in self.R]
        return D

    def project(self, p, plane):
        # Project p onto $a$ random planes
        # plane is norm orthogonal vector to plane
        return p - np.dot(p, plane) * plane

    # Actually returns the norm orthogonal vector to the plain
    def make_planes(self, dim):
        planes = []
        for _ in range(self.a):
            b = np.random.sample(size = dim)
            b = b/np.linalg.norm(b)
            signs = np.random.choice( (-1, 1), size=dim)
            b = np.multiply(b, signs)
            planes.append(b)
        return planes

    def __repr__(self):
        return str([self.A[r] for r in self.R])

def main():
    '''
    trainFile = sys.argv[1]
    testFile = sys.argv[2]

    traindata = np.loadtxt(trainFile, delimiter=',')
    testdata = np.loadtxt(testFile, delimiter=',')
    y_train, X_train = map(int, traindata[:,0]),traindata[:,1:]
    y_test, X_test = map(int,testdata[:,0]),testdata[:,1:]

    models = dict()
    for fv,y in zip(X_train,y_train):
        try:
            m = models[y]
        except:
            m = Hash()
            models[y] = m 
        m.train(fv, R=[0.01, 0.02, 0.03, 0.04], a=3)

    print models

    for fv,y in zip(X_test,y_test):
        pts = [ len( [m for m in [models[idx].behavior_hash(fv, i) for i in range(1, len(fv))] if m] ) for idx in models ]
        m = float(max(pts))
        #print pts
        probas = [ (m-x)/m for x in pts ]
        try:
            probas = [x/sum(probas) for x in probas]
        except:
            pass
        y_true = [ 1 if i is y-1 else 0 for i in range(len(models)) ]
        #print probas
        print pts
        print y-1
        fpr,tpr,_ = roc_curve(y_true, probas)
        #print auc(fpr,tpr)

    '''
    model = Hash()
    with open(sys.argv[1]) as f:
        t = map(float, f.read().split())
    model.train(t, R=[25, 50, 100], a=5)
    with open(sys.argv[2]) as f:
        t = map(float, f.read().split())
        pts = [m for m in [model.behavior_hash(t, i) for i in range(1, len(t))] if m]
    print pts
    print len(pts)*3/float(len(t))

if __name__ == "__main__":
    main()

