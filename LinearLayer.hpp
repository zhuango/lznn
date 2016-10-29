#include <vector>
#include <cmath>
#include <random>

#include "types.h"

class LinearLayer
{
    public:
        LinearLayer(size_t dataSize, size_t inputSize, size_t outputSize)
        :inputSize(inputSize),outputSize(outputSize)
        {
            this->input  = vector<double>(inputSize, 0);
            this->output = vector<double>(outputSize, 0);
        }
        LinearLayer(vector<double> &input, size_t dataSize, size_t inputSize, size_t outputSize)
        :input(input), W0(vector<double>(outputSize, 1.0)), output(vector<double>(outputSize, 0.0))
        {
            this->dataSize   = dataSize;
            this->inputSize  = inputSize;
            this->outputSize = outputSize;

            std::default_random_engine generator;
            std::uniform_real_distribution<double> distribution(-1.0, 1.0);

            for(size_t i = 0; i < inputSize * outputSize; i++)
            {
                W[i] = distribution(generator);
            } 
        }
        void ForwPropagate()
        {
            double sum;
            for(int i = 0; i < outputSize; i++)
            {
                sum = 0.0;
                for(int j = 0; j < inputSize; j++)
                {
                    sum += W[i * inputSize + j] * input[j];
                }
                output[i] = avtivition(sum + W0[i]);
            }
        }
        void BackPropagate()
        {
            //TODO: back propagate
        }
        vector<double> *GetOutput()
        {
            return &(this->output);
        }

        vector<double> W;
        vector<double> W0;
        size_t dataSize;
        size_t inputSize;
        size_t outputSize;
        vector<double> input;
    private:
        vector<double> output;
        double avtivition(double in)
        {
            double result = 1.0 / (1.0 + exp(-in));
            return result;
        }
        vector<double> delta;
};
