#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<random>
#include<ctime>
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

//Debug
    // std::default_random_engine generator;
    // std::uniform_real_distribution<double> dis(-2, 2);
    // for(int i = 0;i < 20; i ++)
    // {
    //     inputs[0][i] = dis(generator);
    // }
//Debug

    size_t dataSize =5000;//

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

    clock_t startForw;
    clock_t startBack;
    clock_t forwDone;
    clock_t backDone;

    for (int j = 0; j < 2000; j++)
    {
        //performance/////////////
        startForw = clock();
        //performance/////////////
        mlp.ForwPropagate();
        //performance/////////////
        forwDone = clock();
        cout << "forward: " << float(forwDone - startForw) / CLOCKS_PER_SEC << " s." << endl;
        //performance/////////////

        Matrix *output = mlp.Output();
        int currectCounter = 0;
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
        //cout << "iter: " << j << " " << double(double(currectCounter) / double(dataSize)) << endl;

        //performance/////////////
        startBack = clock();
        //performance/////////////
        mlp.BackPropagate(1.0/double(dataSize));
        //performance/////////////
        backDone = clock();
        cout << "backward: " << float(backDone - startBack) / CLOCKS_PER_SEC << " s." << endl;
        //performance/////////////
        
        mlp.CleanGredient();
    }
    
    return 0;
}