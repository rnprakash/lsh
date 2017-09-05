#ifndef __lsh_H__
#define __lsh_H__

#include <functional>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <map>
#include <stdint.h>
#include <random>
#include <numeric>
#include <set>
#include <math.h>
#include <algorithm>
#include <iterator>

/* Simulate for i,v in enumerate(vec): in python */
//#define enumerate( vec ) std::size_t i = 0; for( auto v = vec.begin(); v != vec.end(); v++, i++ )
#define range( n ) boost::irange(0u,(uint32_t)n)

//template< typename double >
class lsh {
    private:
        typedef std::vector<int32_t> intvec;
        typedef std::vector<uint32_t> uintvec;
        typedef std::vector<double> fvec;
        typedef std::vector< double > dvec;

        std::function<double(dvec,dvec)> dot = [&](dvec v1, dvec v2) -> double { return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0); };
        std::function<dvec(dvec,dvec)> diff = [&](dvec v1, dvec v2) -> dvec {
            dvec res;
            std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(res), [&](double f, double s) { return f-s; } );
            return res;
        };
        std::function<dvec(dvec,uint32_t)> scale = [&](dvec v, uint32_t s) -> dvec {
            std::transform(v.begin(), v.end(), v.begin(), [&](double e) { return e*s; } );
            return v;
        };

        /* Feature map function */
        dvec feature_map(dvec v)
        {
           dvec ediff;
           std::transform(v.begin()+1, v.end(), v.begin(), std::back_inserter(ediff),
                   [this](double f, double s) { return s - f; } );
           v.insert(v.end(), ediff.begin(), ediff.end());
           return v;
        }
        std::function<dvec(dvec)> phi = [=](dvec v) { return feature_map(v); };

        /* Distance function */
        double dist_function(dvec v1, dvec v2)
        {
            dvec res;
            std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(res),
                    [this](double i, double j) { return (i-j)*(i-j); } );
            return sqrt(std::accumulate(res.begin(), res.end(), 0u));
        }
        std::function<double(dvec, dvec)> dist = [=](dvec v1, dvec v2) { return this->dist_function(v1,v2); } ;

        std::map<double, std::vector< std::set< uint32_t> > > A; // known values dictionary mapping
        fvec R; // Distance sensitivity values
        uintvec K; // Hashing window sizes
        uint32_t M = 1<<16, // Modulus int
                 a = 3, // Number of hash functions per projection
                 b = 4; // Number of hash bands
        std::vector< std::vector< std::vector < fvec > > > planes; // list of random planes planes per band per window
        std::vector< std::pair<dvec, dvec> > proj_axis;

        bool is_anomalous(std::vector< uintvec > distances);
        std::vector< uintvec > hash(dvec tk, uint32_t k_idx);
        dvec project(dvec point, fvec plane);
        std::vector< fvec > make_planes(uint32_t, std::mt19937&, std::uniform_real_distribution<>&);
        void gen_planes();
        void init();
        void make_proj_mat( dvec ts );

    public:
        lsh( fvec R, uintvec K ) : R(R), K(K) { init(); };
        lsh( fvec R, uintvec K, uint32_t a, uint32_t b ) : R(R), K(K), a(a), b(b) { init(); };
        lsh( std::function<dvec(dvec)> phi,
             std::function<double(dvec, dvec)> d,
             fvec R,
             uintvec K,
             uint32_t a,
             uint32_t b
           ) : phi(phi), dist(dist), R(R), K(K), a(a), b(b) { init(); };
        // feature map, distance function, sensitivies, window sizes, hash functions, bands
        ~lsh();
        std::vector< std::pair<uint32_t, uint32_t> > behavior_hash( dvec );
        void train( dvec );
        void train_data( std::string );
        std::vector< std::vector< std::pair<uint32_t, uint32_t> > > test_data( std::string );
        void clear();
};

#endif//__lsh_H__
