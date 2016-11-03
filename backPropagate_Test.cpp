#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>

using namespace std;

#include "LinearLayer.hpp"
#include "lznn_types.h"
#include "MLP.hpp"

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

void fillData(string labelFile, vector<int> &labels)
{
    int a = 0;
    ifstream labelF;
    labelF.open(labelFile.c_str());
    unsigned int counter = 0;
    while(!labelF.eof())
    {
        labelF >> a;
        labels.push_back(a - 1);
    }
}

int main(void)
{
    Matrix inputs; 
    Matrix W1;
    Matrix W2;
    VectorInt labels;
    
    fillData("./data/X.txt", inputs);
    fillData("./data/Theta1.txt", W1);
    fillData("./data/Theta2.txt", W2);
    fillData("./data/label.txt", labels);

    size_t dataSize = 5000;

    VectorInt layersNumbers;
    layersNumbers.push_back(400);
    layersNumbers.push_back(25);
    layersNumbers.push_back(10);

    MLP mlp(&inputs, &labels, 1, layersNumbers);
    for (int j = 0; j < 50; j++)
    {
        for(int i = 0; i < dataSize; i++)
        {
            Matrix input(1, inputs[i]);
            VectorInt label(1, labels[i]);
            mlp.SetInput(&input, &label, 1);
            mlp.ForwPropagate();
            mlp.BackPropagate(0.1);
            mlp.CleanGredient();
        }    

        int currectCounter = 0;
        for(size_t i = 0; i < dataSize; i++)
        {
            Matrix input(1, inputs[i]);
            VectorInt label(1, labels[i]);
            mlp.SetInput(&input, &label, 1);
            mlp.ForwPropagate();
            Matrix *output = mlp.Output();

            size_t predict = 0;
            for (size_t j = 0; j < 10; j++)
            {
                //cout << (*output)[0][j] << " ";
                if ((*output)[0][j] > (*output)[0][predict])
                {
                    predict = j;
                }
            }
            //cout << endl;
            //cout << predict << " " << labels[i] << endl;//////////
            if (predict == labels[i])
            {
                currectCounter += 1;
            }
        }
        cout << "iter: " << j << " " << double(currectCounter / 5000.0) << endl;
    }

    
    return 0;
}