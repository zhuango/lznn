#ifndef _COMMON_H_
#define _COMMON_H_

#include<fstream>
#include<sstream>
#include<vector>

#include "../src/lznn_PerceptronLayer.cpp"
#include "../src/lznn_types.h"
#include "../src/lznn_MLP.cpp"
#include "../src/lznn_tools.cpp"


vector<string> *split(string line, char delim)
{
    istringstream toLine(line);
    vector<string> *tokens = new vector<string>();
    string item;
    while(getline(toLine, item, delim))
    {
        tokens->push_back(item);
    }
    return tokens;
}
double stringToDouble(string str)
{
    istringstream tran(str);
    double a = 0;
    tran >> a;
    return a;
}
void fillData(string dataFile, Matrix &data)
{
    int a = 0;
    ifstream dataF;
    dataF.open(dataFile.c_str());
    string oneLine;
    while(getline(dataF, oneLine))
    {
        Vector tmp;
        vector<string> *number = NULL;
        number = split(oneLine,' ');
        for(int i = 0; i < number->size(); i++)
        {
            tmp.push_back(stringToDouble((*number)[i]));
        }
        data.push_back(tmp);
        delete number;
    }
    dataF.close();
}

void fillData(string labelFile, vector<int> &labels, bool zeroBaseIndex = false)
{
    int a = 0;
    ifstream labelF;
    labelF.open(labelFile.c_str());
    unsigned int counter = 0;
    while(!labelF.eof())
    {
        labelF >> a;
        if (zeroBaseIndex)
        {
            labels.push_back(a);
        }
        else
        {
            labels.push_back(a - 1);
        }
    }
}


#endif