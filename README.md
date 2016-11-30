Consider $n$ streams of time series describing the utilization traces of
$n$ resources, each at some (fixed) granularity. Denote the space of a
time series as $S$

Define a function $d$:

$$\begin{aligned}
    d_{a,b} \colon S^2 &\to [0,1]\\
    (x[a:b],y[a:b]) &\mapsto [0,1] {\addtocounter{equation}{1}\tag{\theequation}}\end{aligned}$$

where $d$ maps a portion of two time series to a real number in $[0,1]$
that represents the distance between the partial time series.

Additionally, we require $d$ to satisfy the triangle inequality:

$$\begin{aligned}
    d(x,y) \le d(x,z) + d(z,y) {\addtocounter{equation}{1}\tag{\theequation}}\end{aligned}$$

Out of the box Dynamic Time Warping (DTW) does not satisfy the triangle
inequality, so we leverage an approximate lower-bound
DTW \[https://arxiv.org/pdf/0807.1734.pdf\]
