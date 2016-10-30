#include <vector>
#include <cmath>
#include <random>

#include "lznn_types.h"
#include "lznn_math.hpp"

class LinearLayer
{
    public:

        Matrix W;
        Vector W0;

        LinearLayer(size_t dataSize, size_t inputSize, size_t outputSize)
        :LinearLayer(Matrix(dataSize, Vector(inputSize, 0.0)), dataSize, inputSize, outputSize)
        {
        }
        LinearLayer(Matrix input, size_t dataSize, size_t inputSize, size_t outputSize)
        :
            input(input),
            output(Matrix(dataSize, Vector(outputSize, 0.0))), 
            W0(Vector(outputSize, 0.0)),
            W(Matrix(outputSize, Vector(inputSize, 0.0)))
        {
            this->dataSize   = dataSize;
            this->inputSize  = inputSize;
            this->outputSize = outputSize;
            initW();
        }

        void ForwPropagate()
        {
            double sum;
            for (int k = 0; k < dataSize; k++)
            {
                for(int i = 0; i < outputSize; i++)
                {
                    sum = 0.0;
                    for(int j = 0; j < inputSize; j++)
                    {
                        sum += W[i][j] * input[k][j];
                    }
                    output[k][i] = avtivition(sum + W0[i]);
                }
            }
        }

        void BackPropagate()
        {
            //TODO: back propagate
        }

        Matrix *Output()
        {
            return &(this->output);
        }

        void SetInput(Matrix &intput)
        {
            this->input = intput;
        }
    private:

        Matrix input;
        Matrix output;
        Matrix delta;
        size_t dataSize;
        size_t inputSize;
        size_t outputSize;

        double avtivition(double in)
        {
            double result = 1.0 / (1.0 + exp(-in));
            return result;
        }
        void initW()
        {
            std::default_random_engine generator;
            std::uniform_real_distribution<double> distribution(-1.0, 1.0);
            for (size_t i = 0; i < this->outputSize; i++)
            {
                for (size_t j = 0; j < this->inputSize; j++)
                {
                    W[i][j] = distribution(generator);
                }
                W0[i] = distribution(generator);
            }
        }
};
