#! /usr/bin/python

import sys
import numpy as np
import random as rand
from collections import defaultdict
import pickle
import argparse
import dill

# Training:
#   - Determine PCA projection for input set for each window size
#   - Create $a$ random lines
#   - For each band, for each of $a$ lines, project PCA'd point onto line,
#       determine min bucket per band per threshold.
#       Store values in dictionary
# Testing:
#   - Perform PCA projection on input time series based on training covariance
#   - For each band, for each of $a$ lines, project PCA'd point onto line,
#       determine min bucker per band per threshold.
#       Compare values with dictionary to determine behavior type

def constant(ts):
    return ts

class Hash:
    def deriv(ts):
        return np.ediff1d(np.array(ts))

    def feature_map(ts):
        return np.concatenate((ts, np.ediff1d(np.array(ts))))

    def distance(x,y):
        return np.linalg.norm(x-y)


    # \phi is feature mapping, d is distance function
    def __init__(self, phi=feature_map, d=distance):
        self.phi = phi
        self.d = d
        self.A = defaultdict( lambda: defaultdict(set) )
        self.cov = dict()
        self.ts = defaultdict(list)
        self.planes = None

    def delte_model(self):
        self.A.clear()
        self.ts.clear()
        self.cov.clear()

    def train(self, t, R, K = [3,5], a = 5, b = 3,  M = 2**16):
        self.R = R
        self.K = K
        self.a = a
        self.M = M
        self.b = b
        if self.planes is None:
            self.planes = self.make_planes(b)

        for k in K:
            for i in range(0, len(t), k/2):
                tk = t[i:i+k]
                if len(tk) is not k:
                    continue
                D = self.hash(tk, k)
                for d,r in zip(D,self.R):
                    for b,v in enumerate(d):
                        self.A[r][b].add(v)
        return sum(len(self.A[k]) for k in self.A)

    # t is a real-time time series
    # behaviors is true if known for k, false if anomalous, None otherwise
    def behavior_hash(self, t):
        behaviors = []
        for k in self.K:
            for i in range(0, len(t), k/2):
                tk = t[i:i+k]
                if len(tk) is not k:
                    continue
                # D, indexed by r \in R, is distance per units in R
                D = self.hash(tk, k)
                # Check if known
                if self.is_known(D):
                    behaviors.append((i,k))
        return behaviors

    def is_known(self, D):
        # A[r] is dictionary of known values for threshold r. 0 is known, 1 is not
        # known is 2D list of behavior values
        known = [ 1 if all( d[b] in self.A[r][b] for b in range(self.b) ) else 0 for d,r in zip(D,self.R) ]
        #return sum(known) > len(known)/2.
        return all(known)

    def is_anomalous(self, D):
        # A[r] is dictionary of known values for threshold r. 0 is known, 1 is not
        # known is 2D list of behavior values
        anomalous = [ 0 if ( all( d[b] in self.A[r][b] for b in range(self.b) ) ) else 1 for d,r in zip(D,self.R) ]
        thresholds = [ sum( anomalous[i:] ) for i in range(len(anomalous)/2) ]
        return any( thresholds[i] > (len(thresholds) - i) for i in range(len(thresholds)) )

    # Hash time series with window length w
    # Window length is a variable parameter that defines the sensitivity of the hash
    def hash(self, tk, k):
        # Compute feature map on tk
        p = np.array(self.phi(tk))
        # Embed p in 2d with random projection
        # Project p onto $a$ random matrices
        planes = self.planes[k]
        P = [ [self.project(p, plane) for plane in planes[b] ] for b in range(self.b) ]
        # Get minimum distance from origin of points per unit r for each band
        distances = [ np.array([ self.d(0, proj) for proj in P[b] ]) for b in range(self.b) ]
        D = [ [ int(min(d/r % self.M)) for d in distances ] for r in self.R]
        return D

    def project(self, p, plane):
        # Project p onto $a$ random matrices
        return np.dot(p,plane)

    # Return matrices of size (K, 2) for each band, for each hash
    def make_planes(self, dim):
        planes = {}
        for k in self.K:
            p = []
            for _ in range(dim):
                b = [ np.random.randn(k,2) for _ in range(self.a) ]
                b = [ m/np.linalg.norm(m) for m in b ]
                p.append(b)
            planes[k] = p
        return planes

    def __repr__(self):
        return str([self.A[r] for r in self.R])

def main():
    parser = argparse.ArgumentParser(description='Train and test LSH-based models')
    parser.add_argument('--train', type=str, nargs='+', required=False,
                        help='Input file containing training sequence fragments')
    parser.add_argument('--test', type=str, nargs='+', required=False,
                        help='Input file containing testing sequence fragments')
    parser.add_argument('--model', type=str, help='Filename of model to use. If exists, will be loaded, else created')
    parser.add_argument('-K', type=int, nargs='+', required=False,
                        default=[50], help='Hashing windows')
    parser.add_argument('-R', type=float, nargs='+', required=False,
                        default=[0.1], help='Ranking sensitivity')
    args = parser.parse_args()

    # Train on file
    model = None
    if args.train:
        model = Hash(phi=constant)
        for fname in args.train:
            trace = np.loadtxt(fname, delimiter=',')
            for t in trace:
                model.train(t, R=args.R, K=args.K)
        # Dump np model if trained
        if args.model:
            with open(args.model, 'wb') as f:
                pickle.dump(model, f)
    # Test on testfile
    if args.test:
        if not model:
            with open(args.model, 'rb') as f:
                model = pickle.load(f)
        for fname in args.test:
            print fname
            trace = np.loadtxt(fname, delimiter=',')
            known = 0
            for t in trace:
                known += len(model.behavior_hash(t))
            print '%d/%d' % (known,trace.shape[0])

if __name__ == "__main__":
    main()

