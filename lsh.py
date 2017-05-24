#! /usr/bin/python

import sys
import numpy as np
import random as rand
from collections import defaultdict
from sets import Set
from sklearn.metrics import roc_curve, auc
import time

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
        np.random.seed(int(time.time()))

    def delte_model(self):
        self.A.clear()

    def train(self, t, R, K = [3,5], a = 5, b = 3, M = 2**16):
        self.R = R
        self.K = K
        self.a = a
        self.M = M
        self.b = b

        distances = []
        for i in range(len(t)):
            if i % self.K[0] != 0 and i % self.K[0] != self.K[0]/2:
                continue
            for k in self.K:
                tk = t[i:i+k]
                D = self.hash(tk)
                distances.append(self.d(tk, np.zeros(len(tk))))
                for d,r in zip(D,self.R):
                    for b,v in enumerate(d):
                        try:
                            self.A[r][b].add(v)
                        except:
                            self.A[r][b] = Set([v])
        return distances

    # t is a real-time time series
    # i is index to current time
    # behaviors is true if known for k, false if anomalous, None otherwise
    def behavior_hash(self, t, i):
        behaviors = []
        if i % self.K[0] != 0 and i % self.K[0] != self.K[0]/2:
            return []
        for k in self.K:
            tk = t[i:i+k]
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
        return any( thresholds[i] > (len(thresholds) - i) for i in range(len(thresholds)) )

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
        D = [ [ int(min( d/r ) ) for d in distances ] for r in self.R]
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
    trainFile = sys.argv[1]
    testFile = sys.argv[2]

    traindata = np.loadtxt(trainFile, delimiter=',')
    testdata = np.loadtxt(testFile, delimiter=',')
    y_train, X_train = map(int, traindata[:,0]),traindata[:,1:]
    y_test, X_test = map(int,testdata[:,0]),testdata[:,1:]

    #y_train = [y - 1 for y in y_train]
    #y_test = [y - 1 for y in y_test]

    models = dict()
    print "Training model"
    for fv,y in zip(X_train,y_train):
        try:
            m = models[y]
        except:
            m = Hash()
            models[y] = m 
        m.train(fv, R=np.linspace(0.01,0.10,num=20), a=5, K=[9,23,57,111], b=3)

    for fv,y in zip(X_test,y_test):
        pts = [m for m in [models[0].behavior_hash(fv, i) for i in range(1, len(fv))] if m]
        print '%d, %f' % (y, len(pts)/float(len(fv)))
        pts = [ len( [m for m in [models[idx].behavior_hash(fv, i) for i in range(1, len(fv))] if m] ) for idx in range(len(models)) ]
        m = float(max(pts))
        try:
            probas = [ (m-x)/m for x in pts ]
            probas = [x/sum(probas) for x in probas]
        except:
            continue
        y_true = [ 1 if i is y else 0 for i in range(len(models)) ]
        print probas
        print y_true
        print
        #fpr,tpr,_ = roc_curve(y_true, probas)
        #print auc(fpr,tpr)

    '''
    print "Loading time series"
    model = Hash()
    t = np.loadtxt(sys.argv[1], delimiter=',')
    #with open(sys.argv[1]) as f:
    #    t = map(float, f.read().split(','))
    print "Training model"
    model.train(t[:50000], R=[25, 50, 100], a=5, b=3, K=[50,100])
    #with open(sys.argv[2]) as f:
    #    t = map(float, f.read().split())
    print "Loading testing data"
    t = np.loadtxt(sys.argv[2], delimiter=',')
    print "Testing against model"
    pts = [m for m in [model.behavior_hash(t, i) for i in range(1, len(t))] if m]
    print pts
    print len(pts)*3/float(len(t))
    '''

if __name__ == "__main__":
    main()

