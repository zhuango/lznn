#ifndef _MLP_H_
#define _MLP_H_

#include "lznn_PerceptronLayer.cpp"
#include "lznn_RegularizedPerceptronLayer.cpp"
#include "lznn_types.h"
#include "lznn_tools.cpp"

#include <list>

class MLP
{
    public:
        MLP(Matrix *input, VectorInt *labels, size_t dataSize, VectorInt &sizeOfLayers)
        :
            deltas(Matrix(dataSize, Vector(sizeOfLayers[sizeOfLayers.size() - 1], 0.0)))
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
                this->layers.push_back(PerceptronLayer(dataSize, sizeOfLayers[i - 1], sizeOfLayers[i]));
                i += 1;
            }
        }        
        MLP(Matrix *input, VectorInt *labels, size_t dataSize, VectorInt &sizeOfLayers, double regularizationCoefficient)
        :
            deltas(Matrix(dataSize, Vector(sizeOfLayers[sizeOfLayers.size() - 1], 0.0)))
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
                this->layers.push_back(RegularizedPerceptronLayer(dataSize, sizeOfLayers[i - 1], sizeOfLayers[i], regularizationCoefficient));
                i += 1;
            }
        }
        void ForwPropagate()
        {
            Matrix *curInput = this->input;
            int i = 0;
            while (i < this->numberOfLayer)
            {
                layers[i].SetInput(curInput);
                layers[i].ForwPropagate();
                curInput = layers[i].Output();
                i += 1;
            }
            this->output = this->layers[numberOfLayer - 1].Output();
        }
        
        void BackPropagate(double learningRate)
        {
            calculateDeltas();
            //Debug////////////////
            // Tools::dump(*(layers[numberOfLayer - 1].GetW()), "W_" + Tools::ToString(numberOfLayer - 1), "W");
            // Tools::dump(*(layers[numberOfLayer - 1].GetW0()), "W0_" + Tools::ToString(numberOfLayer - 1), "W0");
            //Debug////////////////
            layers[numberOfLayer - 1].NextLinearWeightDelta = &(this->deltas);
            layers[numberOfLayer - 1].BackPropagate(learningRate);
            //Debug////////////////
            // Tools::dump(*(layers[numberOfLayer - 1].NextLinearWeightDelta), "NextLinearWeightDelta_" + Tools::ToString(numberOfLayer - 1), "NextLinearWeightDelta");
            // Tools::dump(*(layers[numberOfLayer - 1].Input()), "input_" + Tools::ToString(numberOfLayer - 1), "input");
            // Tools::dump(*(layers[numberOfLayer - 1].Output()), "output_" + Tools::ToString(numberOfLayer - 1), "output");
            // Tools::dump(*(layers[numberOfLayer - 1].GetGredientW()), "gredientW_" + Tools::ToString(numberOfLayer - 1), "GredientW");
            // Tools::dump(*(layers[numberOfLayer - 1].GetGredientW0()), "gredientW0_" + Tools::ToString(numberOfLayer - 1), "GredientW0");
            // Tools::dump(*(layers[numberOfLayer - 1].GetW()), "Back_W_" + Tools::ToString(numberOfLayer - 1), "W");
            // Tools::dump(*(layers[numberOfLayer - 1].GetW0()), "Back_W0_" + Tools::ToString(numberOfLayer - 1), "W0");
            // Tools::dump(layers[numberOfLayer - 1].deltas, "deltas" + Tools::ToString(numberOfLayer - 1), "deltas");
            //Debug////////////////
            int i = numberOfLayer - 2;
            while(i >= 0)
            {
                //Debug////////////////
                // Tools::dump(*(layers[i].GetW()), "W_" + Tools::ToString(i), "W");
                // Tools::dump(*(layers[i].GetW0()), "W0_" + Tools::ToString(i), "W0");
                //Debug////////////////
                layers[i].NextLinearWeightDelta = &(layers[i + 1].WeightDelta);
                layers[i].BackPropagate(learningRate);
                //Debug////////////////
                // Tools::dump(*(layers[i].NextLinearWeightDelta), "NextLinearWeightDelta_" + Tools::ToString(i), "NextLinearWeightDelta");
                // Tools::dump(*(layers[i].Input()), "input_" + Tools::ToString(i), "input");
                // Tools::dump(*(layers[i].Output()), "output_" + Tools::ToString(i), "output");
                // Tools::dump(*(layers[i].GetGredientW()), "gredientW_" + Tools::ToString(i), "GredientW");
                // Tools::dump(*(layers[i].GetGredientW0()), "gredientW0_" + Tools::ToString(i), "GredientW0");
                // Tools::dump(*(layers[i].GetW()), "Back_W_" + Tools::ToString(i), "W");
                // Tools::dump(*(layers[i].GetW0()), "Back_W0_" + Tools::ToString(i), "W0");
                // Tools::dump(layers[i].deltas, "deltas" + Tools::ToString(i), "deltas");
                //Debug////////////////
                i -= 1;
            }
        }
        void CleanGredient()
        {
            for(auto &layer: layers)
            {
                layer.CleanGredient();
            }
        }
        Matrix *Output()
        {
            return this->output;
        }
        void SetInput(Matrix *inputs, VectorInt *labels, size_t dataSize)
        {
            this->input    = inputs;
            this->labels   = labels;
            this->dataSize = dataSize;
        }
        // MLP Clone()
        // {

        // }
        vector<PerceptronLayer> layers;
    private:
        Matrix             *input;
        Matrix              deltas;
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
                for(size_t j = 0; j < outputSize; j++)
                {
                    if ((*labels)[i] == j)
                    {
                        deltas[i][j] = 1.0 - ((*output)[i][j]);
                    }
                    else
                    {
                        deltas[i][j] = 0.0 - ((*output)[i][j]);
                    }
                }
            }
        }
        int softmax(Vector *input)
        {
            int max = 0;
            for(size_t i = 0; i < inputSize; i++)
            {
                if ((*input)[i] > (*input)[max])
                {
                    max = i;
                }
            }
            return max;
        }
};

#endif