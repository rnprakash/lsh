#include <boost/range/irange.hpp>
#include <boost/foreach.hpp>
#include <boost/range/combine.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "lsh.h"

void lsh::init()
{
    // Generate random planes (by their normal)
    gen_planes();
    // Initialize dictionary of values
    for( double r : R)
        A[r] = std::vector< std::set< uint32_t > >(b);
}

lsh::~lsh()
{
    clear();
}

// Generate random planes (by their normal)
void lsh::gen_planes()
{
    std::uniform_real_distribution<> r(0,1);
    std::random_device rd;
    std::mt19937 m(rd());
    for( auto k : K )
    {
        std::vector< std::vector< fvec > > window_planes;
        for( auto i : range(b) )
        {
            // Generate random k-dim plane by normal
            window_planes.push_back( make_planes(k, m, r) );
        }
        planes.push_back( window_planes );
    }
}

/* $b \cross a$ planes for each dim */
std::vector< std::vector<double> > lsh::make_planes(uint32_t dim, std::mt19937& m, std::uniform_real_distribution<>& r)
{
    std::vector< fvec > ret_planes;
    /* Each window length $k$ has different set of planes
     * with $b$ bands. Each band holds $a$ planes.
     */
    for( auto i : range(a) )
    {
        dvec a_planes;
        std::transform( range(dim).begin(), range(dim).end(), std::back_inserter(a_planes),
                        [this, &m, &r](uint32_t i) { return r(m); }
                      );
        ret_planes.push_back(a_planes);
    }
    return ret_planes;
}

/* 
 * Delete model to be retrained
 * Keeps R, K, a, and b the same
 */
void lsh::clear()
{
    A.clear();
    planes.clear();
}

/*
 * Determine whether calculated distance hashes are anomalous or not
 * vector< vector < uint > > distances, vector of min distances per band, per threshold
 *         [ [ band i mindist for threshold r , ... for band i in  bands ] , ... for r in R ]
 * returns true if < 1/2 of thresholds have seen the distances
 *         false otherwise
 */
bool lsh::is_anomalous(std::vector< uintvec > distances)
{
    uintvec anomalies;
    for( auto i : range(R.size()) )
    {
        uintvec counts;
        // Determine set inclusion for threshold r over each band
        std::transform(distances[i].begin(), distances[i].end(), A[R[i]].begin(), std::back_inserter(counts),
            [this](uint32_t d, std::set<uint32_t> ar_d) { return ar_d.count(d); } );
        anomalies.push_back( ( std::accumulate(counts.begin(), counts.end(), 0u) == b ) ? 1 : 0 );
    }
    // Determine if any >1/2 size slice has more anomalies than the set size
    uintvec thresholds;
    std::transform(range(R.size()).begin(), range(R.size()).begin() + R.size()/2, std::back_inserter(thresholds),
                   [this, &anomalies](uint32_t idx) { return std::accumulate( anomalies.begin() + idx, anomalies.end(), 0u); } );
    return std::any_of( range(thresholds.size()).begin(), range(thresholds.size()).end(),
                        [this, &thresholds](uint32_t idx) { return thresholds[idx] > (thresholds.size() - idx); } );
}

/*
 * Determine projection onto two dimensions by top two axes by covariance
 */
void lsh::make_proj_mat( dvec ts )
{
    // Calculate covariance matrix for each k
    for( uint32_t k : K )
    {
        std::vector< dvec > segments;
        for( std::size_t i = 0; i < ts.size(); i += K[0]/2 )
        {
            dvec seg(ts.begin() + i, ts.begin() + i + k);
            if( seg.size() == k )
                segments.push_back(seg);
        }
        // Get covariance matrix for top two projection vectors
        cv::Mat m(segments);
        cv::Mat cov, mean;
        cv::calcCovarMatrix(&m, (int)segments.size(), cov, mean, CV_COVAR_SCALE, CV_64F);
        cov = cov.t(); // Transpose to get rows instead of cols
        dvec data(*(double*)cov.data);
        dvec first(data.begin(), data.begin() + k);
        dvec second(data.begin() + k + 1, data.begin() + k + k);
        proj_axis.push_back( std::pair<dvec,dvec>(first,second) );
    }

}

/*
 * Quantify seen behavior of ts and determine when and where anomalies occur
 * dvec ts, time series of observations
 * returns vector of (uint, uint) pairs: (index, observation window) of observed anomaly
 */
std::vector< std::pair<uint32_t, uint32_t> > lsh::behavior_hash(dvec ts)
{
    assert( A.size() != 0 );
    
    std::vector< uintvec > behaviors( ts.size() * 2 / K[0] + 1, uintvec() );

    for(std::size_t i = 0; i < ts.size(); i += K[0] / 2)
    {
        auto& ivec = behaviors[i];
        for( auto it = K.begin(); it != K.end(); it++ )
        {
            uint32_t k = *it;
            if( i + k >= ts.size() )
                continue;
            // tk is subvector ts[i:i+k]
            dvec tk(ts.begin() + i, ts.begin() + i + k);
            // Hash subvector - returns per-band distances for each threshold
            std::vector< uintvec > D = hash(tk, it - K.begin());
            // Compare to known behaviors
            // TODO
            uintvec distances;
            double threshold;
            // For each threshold, add distance to known-value map
            BOOST_FOREACH(boost::tie(distances, threshold), boost::combine(D, R))
            {
                // distances is uintvec of min distances per band for threshold
                // add known distances to dictionary of known distances
                for( auto dt = distances.begin(); dt != distances.end(); dt++)
                {
                    uint32_t idx = dt - distances.begin();
                    //A[threshold][idx].insert(*dt);
                    if( not A[threshold][idx].count(*dt) )
                        ivec.push_back( ( i, k ) );
                }
            }
            //ivec.push_back(  );
        }

    }
}

/*
 * dvec ts, time series as a vector to train on
 * builds dictionary of known values for each interval at eaceh threshold range
 */
void lsh::train( dvec ts )
{
    if(planes.size() == 0)
        gen_planes();
        make_proj_mat(ts);
    // Perform hashing calculations at intervals of K[0] / 2
    for(std::size_t i = 0; i < ts.size(); i += K[0] / 2)
    {
        for( auto it = K.begin(); it != K.end(); it++ )
        {
            uint32_t k = *it;
            // Ensure no out of bounds operations
            if( i + k >= ts.size() )
                continue;
            // tk is subvector ts[i:i+k]
            dvec tk(ts.begin() + i, ts.begin() + i + k);
            // Hash subvector - returns per-band distances for each threshold
            std::vector< uintvec > D = hash(tk, it - K.begin());

            uintvec distances;
            double threshold;
            // For each threshold, add distance to known-value map
            BOOST_FOREACH(boost::tie(distances, threshold), boost::combine(D, R))
            {
                // distances is uintvec of min distances per band for threshold
                // add known distances to dictionary of known distances
                for( auto dt = distances.begin(); dt != distances.end(); dt++)
                {
                    uint32_t idx = dt - distances.begin();
                    A[threshold][idx].insert(*dt);
                }
            }
        }
    }
}

/*
 * dvec tk, window of time series to be hashed
 * k_idx, index of window size (of K)
 * returns std::vector< uintvec >, vector of min distances per band, per threshold
 *         [ [ band i mindist for threshold r , ... for band i in  bands ] , ... for r in R ]
 */
std::vector< std::vector<uint32_t> > lsh::hash( dvec tk, uint32_t k_idx )
{
    dvec p = phi(tk); // Feature map of t_k 
    // Project p onto $a$ random hyperplanes for each band $b$
    std::vector< std::vector < dvec > > kplanes = planes[k_idx];
    std::vector< std::vector< dvec > > projections;
    for( auto& bplanes : kplanes ) // project p onto each plane in bplanes (aplane)
    {
        std::vector< dvec > projs;
        std::transform( bplanes.begin(), bplanes.end(), std::back_inserter(projs),
                        // p - <p, aplane> * aplane
                        [this, &p](dvec aplane) { return diff(p, scale(aplane, dot(p, aplane))); }
                      );
        projections.push_back(projs); 
    }
    // Get minimum distances from origin per band, per threshold
    
    // Calculate all distances
    // distances = [ [d(0,p) for p in proj[i] ] for i in range(b) ]
    std::vector < fvec > distances;
    for( auto i : range(b) )
    {
        fvec per_b_dist;
        // Get $a$ distances for band i
        std::transform( projections[i].begin(), projections[i].end(), std::back_inserter(per_b_dist),
                        [this](dvec& v) -> double { return dist(dvec(v.size(), 0), v); }
                      );
        distances.push_back( per_b_dist );
    }

    // For each R, find minimum distance per band
    std::vector < uintvec > min_distances;
    for( auto r : R )
    {
        uintvec per_r_mindist;
        for( auto i : range(b) )
        {
            dvec dists;
            std::transform( distances[i].begin(), distances[i].end(), std::back_inserter(dists),
                            [this, &r](double di) -> uint32_t { return int(di/r) % M; }
                          );
            per_r_mindist.push_back(*std::min_element(dists.begin(), dists.end()));
        }
        min_distances.push_back(per_r_mindist);
    }
    return min_distances;
}

/*
 * Reads timeseries data from file. Assumes comma-separated values,
 * can be changed to other delims?
 * Trains on read data
 */
void lsh::train_data( std::string filename )
{
    std::ifstream f(filename);
    std::string line;
    while( getline(f, line) )
    {
        std::vector< std::string > cols;
        boost::split(cols, line, boost::is_any_of(", "));
        dvec ts;
        std::transform( cols.begin(), cols.end(), std::back_inserter(ts),
                [this]( std::string s ) { return boost::lexical_cast<double>(s); }
                );
        train(ts);
    }
}

std::vector< std::vector< std::pair<uint32_t, uint32_t> > > lsh::test_data( std::string filename )
{
    std::vector< std::vector< std::pair<uint32_t, uint32_t> > > behaviors;

    std::ifstream f(filename);
    std::string line;
    while( getline(f, line) )
    {
        std::vector< std::string > cols;
        boost::split(cols, line, boost::is_any_of(", "));
        dvec ts;
        std::transform( cols.begin(), cols.end(), std::back_inserter(ts),
                [this]( std::string s ) { return boost::lexical_cast<double>(s); }
                );
        behaviors.push_back( behavior_hash(ts) );
    }
    return behaviors;
}

int main()
{
    lsh l({0.1,0.5}, {5,7});
    return 0;
}
