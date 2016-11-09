#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>

using namespace std;

#include "../LinearLayer.cpp"
#include "../lznn_types.h"
#include "../MLP.cpp"

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

    size_t dataSize = 1;//

    VectorInt layersNumbers;
    layersNumbers.push_back(400);
    layersNumbers.push_back(25);
    layersNumbers.push_back(10);

    MLP mlp(&inputs, &labels, dataSize, layersNumbers);

    // for(size_t i = 0; i < 25; i++)
    // {
    //     mlp.layers[0].W0[i] = W1[i][0];

    //     for(size_t j = 1; j <= 400; j++)
    //     {
    //         mlp.layers[0].W[i][j - 1] = W1[i][j];
    //     }
    // }
    // for(size_t i = 0; i < 10; i++)
    // {
    //     mlp.layers[1].W0[i] = W2[i][0];
    //     for(size_t j = 1; j <= 25; j++)
    //     {
    //         mlp.layers[1].W[i][j - 1] = W2[i][j];
    //     }
    // }


    for (int j = 0; j < 2000; j++)
    {
        mlp.ForwPropagate();

        Matrix *output = mlp.Output();

        int currectCounter = 0;
        for(size_t i = 0; i < dataSize; i++)
        {
            size_t predict = 0;
            for (size_t j = 0; j < 10; j++)
            {
                //cout << (*output)[0][j] << " ";
                if ((*output)[i][j] > (*output)[i][predict])
                {
                    predict = j;
                }
                //cout << (*output)[i][j] << " ";
            }
            //cout << endl;
            if (predict == labels[i])
            {
                currectCounter += 1;
            }
            //cout << predict << " " << labels[i] << " " << currectCounter << endl;//////////
        }
        cout << "iter: " << j << " " << double(double(currectCounter) / double(dataSize)) << endl;

        mlp.BackPropagate(0.1);
        //mlp.Update(0.1);
        mlp.CleanGredient();
    }
    
    return 0;
}