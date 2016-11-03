#include "LinearLayer.hpp"
#include "lznn_types.h"
#include <list>

class MLP
{
    public:
        MLP(Matrix *input, VectorInt *labels, size_t dataSize, VectorInt sizeOfLayers)
        :
            deltas(Vector(sizeOfLayers[sizeOfLayers.size() - 1], 0.0))
        {
            this->input         = input;
            this->labels        = labels;
            this->numberOfLayer = sizeOfLayers.size() - 1;
            this->dataSize      = dataSize;
            this->inputSize     = sizeOfLayers[0];
            this->outputSize    = sizeOfLayers[numberOfLayer];

            size_t i = 1;
            while (i <= numberOfLayer)
            {
                this->layers.push_back(LinearLayer(dataSize, sizeOfLayers[i - 1], sizeOfLayers[i]))
            }
            layers[0].SetInput(input);
        }
        void ForwPropagate()
        {
            for (auto *layer: layers)
            {
                layer.forwPropagate();
            }
            this->output = this->layers[numberOfLayer - 1].Output();
        }
        
        void BackPropagate(double learningRate)
        {
            calculateDeltas();

            layers[numberOfLayer - 1].NextLinearWeightDelta = &(this->delta);
            layers[numberOfLayer - 1].BackPropagate(learningRate);
            
            size_t i = numberOfLayer - 2;
            while(i >= 0)
            {
                layers[i].NextLinearWeightDelta = layers[i + 1].WeightDelta;
                layers[i].BackPropagate(learningRate);
            }
        }
        Matrix *Output()
        {
            return this->output;
        }
    private:
        vector<LinearLayer> layers;
        Matrix             *input;
        Vector              deltas;
        VectorInt          *labels;
        Matrix             *output;

        size_t dataSize;
        size_t inputSize;
        size_t outputSize;
        size_t numberOfLayer;

        void calculateDeltas()
        {
            for(size_t i = 0; i < dataSize; i++)
            {
                predict = softmax((*output)[i]);
                for(size_t j = 0; j < outputSize; j++)
                {
                    if ((*labels)[j] == predict)
                    {
                        deltas[j] += (*output)[i] - 1;
                    }
                    else
                    {
                        deltas[j] += (*output)[i] - 0; 
                    }
                }
            }
        }
        int softmax(Vector *input)
        {
            int max = 0;
            for(size_t i = 0; i < inputSize; i++)
            {
                if ((*input)[i] > (*intput)[max])
                {
                    max = i;
                }
            }
            return max;
        }
};
