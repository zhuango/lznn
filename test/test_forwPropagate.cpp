#include<iostream>

using namespace std;

#include "common.h"

int main(void)
{
    Matrix inputs; 
    Matrix W1;
    Matrix W2;
    VectorInt labels;
    
    fillData("../data/X.txt", inputs);
    fillData("../data/Theta1.txt", W1);
    fillData("../data/Theta2.txt", W2);
    fillData("../data/label.txt", labels);

    size_t dataSize = 5000;

    PerceptronLayer layer1(&inputs, dataSize, 400, 25);
    Vector *W0 = layer1.GetW0();
    Matrix *W  = layer1.GetW();
    for(size_t i = 0; i < 25; i++)
    {
        (*W0)[i] = W1[i][0];

        for(size_t j = 1; j <= 400; j++)
        {
            (*W)[i][j - 1] = W1[i][j];
        }
    }
    layer1.ForwPropagate();

    PerceptronLayer layer2(layer1.Output(), dataSize, 25, 10);
    W0 = layer2.GetW0();
    W  = layer2.GetW();
    for(size_t i = 0; i < 10; i++)
    {
        (*W0)[i] = W2[i][0];
        for(size_t j = 1; j <= 25; j++)
        {
            (*W)[i][j - 1] = W2[i][j];
        }
    }
    layer2.ForwPropagate();

    int currectCounter = 0;
    Matrix *output = layer2.Output();
    for(size_t i = 0; i < dataSize; i++)
    {
        size_t predict = 0;
        for (size_t j = 0; j < 10; j++)
        {
            if ((*output)[i][j] > (*output)[i][predict])
            {
                predict = j;
            }
        }
        if (predict == labels[i])
        {
            currectCounter += 1;
        }
    }
    cout << "should be : 0.9752" << endl;
    cout << double(currectCounter / 5000.0) << endl;
    
    return 0;
}