#include "lsh.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <boost/any.hpp>
#include <boost/algorithm/string.hpp>

// csv['label'] = [0] or [1], ...
// csv['rows'][0] = [1,2,3] ...
template <typename data>
std::map< std::string, std::map< uint32_t, std::vector<data> > > readcsv(std::string filename)
{
    std::map< std::string, std::map< uint32_t, std::vector<data> > > df;
    std::string line;
    // Read training file
    std::ifstream csv(filename);
    for(std::size_t row = 0; getline(csv, line); row++)
    {
        std::vector< std::string > cols;
        boost::split(cols, line, boost::is_any_of(","));
        df["labels"][row].push_back( boost::lexical_cast<uint32_t>(cols[0]) );
        for( std::size_t i = 1; i < cols.size(); i++)
        {
            df["col"][i].push_back( boost::lexical_cast<double>(cols[i]) );
            df["row"][row].push_back( boost::lexical_cast<double>(cols[i]) );
        }
    }

    return df;
}

void read_files(int argc, char* argv[])
{
    if( argc != 3 )
    {
        std::cerr << "Usage: " << argv[0] << " <training file> <testing file>" << std::endl;
        exit(1);
    }
    auto train = readcsv<double>( argv[1] );
    auto test = readcsv<double>( argv[2] );
}


int main(int argc, char* argv[])
{
    //parse_args(argc, argv);
    lsh l({1.,2.}, {3,4});
    l.train_data("train.csv");
    for( auto& behaviors : l.test_data("test.csv") )
    {
        for( auto& p : behaviors )
        {
            std::cout << p.first << ", " << p.second << std::endl;
        }
    }
    return 0;
}
