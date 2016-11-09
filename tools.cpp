#include "lznn_types.h"
#include <string>
#include <fstream>
#include <iostream>
using namespace std;

class Tools
{
    public:
        static void dump(Matrix m, string filename)
        {
            ofstream f;
            f.open(filename.c_str());
            for(size_t i = 0; i < m.size(); i++)
            {
                for(size_t j = 0; j < m[0].size(); j++)
                {
                    f << m[i][j] << " ";
                }
                f << "\n";
            }
        }
        static void dump(Vector m, string filename)
        {
            ofstream f;
            f.open(filename.c_str());
            for(size_t j = 0; j < m.size(); j++)
            {
                f << m[j] << " ";
            }
            f << "\n";
        }
};