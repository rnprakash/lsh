#! /usr/bin/python

import sys
import numpy as np
import random as rand
from collections import defaultdict

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

    def delte_model(self):
        self.A.clear()

    def train(self, t, R, K = [3], a = 5, M = 2**16):
        self.R = R
        self.K = K
        self.a = a
        self.M = M
        for i in range(len(t)):
            for k in self.K:
                if ((i * 2) % k) != 0:
                    continue
                tk = t[-k:]
                D = self.hash(tk)
                for d,r in zip(D,self.R):
                    try:
                        self.A[r][0].append(d[0])
                        self.A[r][1].append(d[1])
                    except:
                        self.A[r][0] = [d[0]]
                        self.A[r][1] = [d[1]]
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
        known = [ 0 if ( d[0] in self.A[r][0] and d[1] in self.A[r][1] ) else 1 for d,r in zip(D,self.R) ]
        thresholds = [ sum( known[i:] ) for i in range(len(known)) ]
        return any( thresholds[i] > i for i in range(len(thresholds)) )

    # Hash time series with window length w
    # Window length is a variable parameter that defines the sensitivity of the hash
    def hash(self, tk):
        # Compute feature map on tk
        p = self.phi(tk)
        # Project p onto $a$ random hyperplanes
        P = [ self.project(p, plane) for plane in self.make_planes(len(p)) ]
        # Get minimum distance from origin of points per unit r
        distances = np.array([self.d(np.zeros(len(p)), p) for p in P])
        D = [ ( int(min( distances/r )), int(max( distances/r )) ) for r in self.R]
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

def main():
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

