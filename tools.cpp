#ifndef _TOOLS_CPP_
#define _TOOLS_CPP_

#include "lznn_types.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

class Tools
{
    public:
        static void dump(Matrix &m, string filename, string variableName)
        {
            int row = m.size();
            int col = m[0].size(); 
            ofstream f;
            f.open(filename.c_str());
            f << variableName << " " << row << " " << col << endl;
            for(size_t i = 0; i < row; i++)
            {
                for(size_t j = 0; j < col; j++)
                {
                    f << m[i][j] << ",";
                }
                f << "\n";
            }
        }
        static void dump(Vector &v, string filename, string variableName)
        {
            ofstream f;
            f.open(filename.c_str());
            f << variableName << " " << v.size() << endl;
            for(size_t j = 0; j < v.size(); j++)
            {
                f << v[j] << ",";
            }
            f << "\n";
        }
        static void dump(double &v, string filename, string variableName)
        {
            ofstream f;
            f.open(filename.c_str());
            f << variableName << endl;
            f << v << endl;
        }
        static string ToString(int a)
        {
            ostringstream ss;
            ss << a;
            return ss.str();
        }
};

#endif