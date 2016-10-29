#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>

using namespace std;

#include "LinearLayer.hpp"

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
void fillData(string dataFile, vector<double> &data)
{
    int a = 0;
    ifstream dataF;
    dataF.open(dataFile.c_str());
    string oneLine;
    while(getline(dataF, oneLine))
    {
        vector<string> *number = NULL;
        number = split(oneLine,' ');
        for(int i = 0; i < number->size(); i++)
        {
            data.push_back(stringToDouble((*number)[i]));
        }
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
        labels.push_back(a);
    }
}

int main(void)
{
    vector<double> inputs; 
    vector<double> W1;
    vector<double> W2;
    vector<int> labels;
    
    fillData("./data/X.txt", inputs);
    fillData("./data/Theta1.txt", W1);
    fillData("./data/Theta2.txt", W2);
    fillData("./data/label.txt", labels);

    LinearLayer layer1(400, 25);

    for(int i = 0; i < 25; i++)
    {
        layer1.W0.push_back(W1[i * 401]);
        for(int j = 1; j <= 400; j++)
        {
            layer1.W.push_back(W1[i * 401 + j]);
        }
    }

    LinearLayer layer2(25, 10);


    for(int i = 0; i < 10; i++)
    {
        layer2.W0.push_back(W2[i * 26]);
        for(int j = 1; j <= 25; j++)
        {
            layer2.W.push_back(W2[i * 26 + j]);
        }
    }

    int currectCounter = 0;
    for(int i = 0; i < 5000; i++)
    {
        for(int j = 0; j < 400; j ++ )
        {
            layer1.input[j] = inputs[i * 400 + j];
        }
        layer1.Forward();

        vector<double> *layer1Output = layer1.GetOutput();
        for(int j = 0; j < layer1Output->size(); j++)
        {
            layer2.input[j] = (*layer1Output)[j];
            //cout << layer1.output[j] << " ";
        }
        layer2.Forward();
        vector<double> *layer2Output = layer2.GetOutput();
        int predict = 0;
        for(int j = 0; j < 10; j++)
        {
            if ((*layer2Output)[j] > (*layer2Output)[predict])
            {
                predict = j;
            }
        }
        //cout << predict + 1 << " " << labels[i] << endl;
        if (predict + 1 == labels[i])
        {
            currectCounter += 1;
        }
    } 
    cout << double(currectCounter / 5000.0) << endl;
    return 0;
}