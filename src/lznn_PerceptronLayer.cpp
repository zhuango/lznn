#ifndef _PerceptronLayer_H_
#define _PerceptronLayer_H_

#include "lznn_types.h"
#include "lznn_math.cpp"
#include <cmath>
#include <random>
#include <ctime>

class PerceptronLayer
{
    public:
        Matrix *NextLinearWeightDelta;
        Matrix WeightDelta;

        PerceptronLayer(size_t dataSize, size_t inputSize, size_t outputSize)
        :PerceptronLayer(nullptr, dataSize, inputSize, outputSize)
        {
        }
        PerceptronLayer(Matrix *input, size_t dataSize, size_t inputSize, size_t outputSize)
        :
            input(input),
            output(Matrix(dataSize, Vector(outputSize, 0.0))),
            WeightDelta(Matrix(dataSize, Vector(inputSize, 0.0))),
            deltas(Matrix(dataSize, Vector(outputSize, 0.0))),
            W0(Vector(outputSize, 0.0)),
            W(Matrix(outputSize, Vector(inputSize, 0.0))),
            gredientW0(Vector(outputSize, 0.0)),
            gredientW(Matrix(outputSize, Vector(inputSize, 0.0)))
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
                        sum += W[i][j] * (*input)[k][j];
                    }
                    output[k][i] = sigmoid(sum + W0[i]);
                }
            }
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
        void CleanGredient()
        {
            for(int i = 0; i < outputSize; i++)
            {
                for(int j = 0; j < inputSize; j++)
                {
                    this->gredientW[i][j] = 0.0;
                }
                this->gredientW0[i] = 0.0;
            } 
        }
        Matrix *Output()
        {
            return &(this->output);
        }
        Matrix* Input()
        {
            return this->input;
        }
        Matrix* GetW()
        {
            return &(this->W);
        }
        Vector* GetW0()
        {
            return &(this->W0);
        }
        Matrix* GetGredientW()
        {
            return &(this->gredientW);
        }
        Vector* GetGredientW0()
        {
            return &(this->gredientW0);
        }
        void SetInput(Matrix *input)
        {
            this->input = input;
        }
        Matrix deltas;

    protected:
        Matrix W;
        Vector W0;
        size_t inputSize;
        size_t outputSize;

        void calculateGredientW()
        {
            for(size_t i = 0; i < dataSize; i++)
            {
                for(size_t j = 0; j < outputSize; j++)
                {
                    deltas[i][j] = output[i][j] * (1.0 - output[i][j]) * (*NextLinearWeightDelta)[i][j];
                }
                for(size_t j = 0; j < outputSize; j++)
                {
                    for (size_t k = 0; k < inputSize; k++)
                    {
                        this->gredientW[j][k] +=  deltas[i][j] * (*input)[i][k];
                    }
                    this->gredientW0[j] += deltas[i][j];
                }
            }
        }
        void calWeightedDeltas()
        {
            for (size_t i = 0; i < dataSize; i++)
            {
                for (size_t j = 0; j < inputSize; j++)
                {
                    double sum = 0;
                    for (size_t k = 0; k < outputSize; k++)
                    {
                        sum += deltas[i][k] * this->W[k][j];
                    }
                    this->WeightDelta[i][j] = sum;
                }
            }
        }
        void update(double learningRate)
        {            
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    W[i][j] += learningRate * gredientW[i][j];
                }
            }
            for (int i = 0; i < outputSize; i++)
            {
                W0[i] += learningRate * gredientW0[i];
            }
        }

    private:

        Matrix *input;
        Matrix output;
        Matrix gredientW;
        Vector gredientW0;
        size_t dataSize;
        static std::default_random_engine generator;

        double sigmoid(double in)
        {
            double result = 1.0 / (1.0 + exp(-in));
            return result;
        }
        double sigmoidGredient(double in)
        {
            double result = sigmoid(in) * (1.0 - sigmoid(in));
            return result;
        }
        void initW()
        {
            std::uniform_real_distribution<double> distribution(-1.0, 1.0);
            for (size_t i = 0; i < this->outputSize; i++)
            {
                for (size_t j = 0; j < this->inputSize; j++)
                {
                    W[i][j] = distribution(PerceptronLayer::generator);
                }
                W0[i] = distribution(PerceptronLayer::generator);
            }
        }
};

std::default_random_engine PerceptronLayer::generator;


#endif