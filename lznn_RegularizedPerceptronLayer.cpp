#ifndef _RegularizedPerceptronLayer_H_
#define _RegularizedPerceptronLayer_H_

#include "lznn_types.h"
#include "lznn_math.cpp"
#include "lznn_PerceptronLayer.cpp"
#include <cmath>
#include <random>
#include <ctime>

class RegularizedPerceptronLayer: public PerceptronLayer
{
    public:

        RegularizedPerceptronLayer(size_t dataSize, size_t inputSize, size_t outputSize, double regularizationCoefficient)
        :
            PerceptronLayer(nullptr, dataSize, inputSize, outputSize),
            regularizationCoefficient(regularizationCoefficient)
        {
        }

        RegularizedPerceptronLayer(Matrix *input, size_t dataSize, size_t inputSize, size_t outputSize,double regularizationCoefficient)
        :
            PerceptronLayer(input, dataSize, inputSize, outputSize),
            regularizationCoefficient(regularizationCoefficient)
        {
        }

        virtual void BackPropagate(double learningRate)
        {
            //performance/////////
            // clock_t start = clock();
            //performance/////////

            calculateGredientW();

            //performance/////////
            // clock_t calGredient = clock() - start;
            // cout << "calculateGredientW: " << float(calGredient)/CLOCKS_PER_SEC << endl;
            // calGredient = clock();
            //performance/////////

            update(learningRate);
            Regularization(learningRate);
            
            //performance/////////
            // clock_t update = clock() - calGredient;
            // cout << "update: " << float(update)/CLOCKS_PER_SEC << endl;
            // update = clock();
            //performance/////////

            calWeightedDeltas();

            //performance/////////
            // clock_t calWeightedDelta = clock() - update;
            // cout << "calWeightedDelta: " << float(calWeightedDelta)/CLOCKS_PER_SEC << endl;
            //performance/////////
        }
        void Regularization(double learningRate)
        {
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    W[i][j] += learningRate * regularizationCoefficient * W[i][j];
                }
            }
            for (int i = 0; i < outputSize; i++)
            {
                W0[i] += learningRate * regularizationCoefficient * W0[i];
            }
        }
    private:
        double regularizationCoefficient;
};

#endif