#include "LinearLayer.cpp"
#include "lznn_types.h"
#include "tools.cpp"

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
                this->layers.push_back(LinearLayer(dataSize, sizeOfLayers[i - 1], sizeOfLayers[i]));
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

            layers[numberOfLayer - 1].NextLinearWeightDelta = &(this->deltas);
            layers[numberOfLayer - 1].BackPropagate(learningRate);
            //Debug////////////////
            Tools::dump(*(layers[numberOfLayer - 1].NextLinearWeightDelta), "NextLinearWeightDelta_" + Tools::ToString(numberOfLayer - 1), "NextLinearWeightDelta");
            Tools::dump(*(layers[numberOfLayer - 1].GetW()), "W_" + Tools::ToString(numberOfLayer - 1), "W");
            Tools::dump(*(layers[numberOfLayer - 1].GetW0()), "W0_" + Tools::ToString(numberOfLayer - 1), "W0");
            Tools::dump(*(layers[numberOfLayer - 1].Input()), "input_" + Tools::ToString(numberOfLayer - 1), "input");
            Tools::dump(*(layers[numberOfLayer - 1].GetGredientW()), "gredientW_" + Tools::ToString(numberOfLayer - 1), "GredientW");
            Tools::dump(*(layers[numberOfLayer - 1].GetGredientW0()), "gredientW0_" + Tools::ToString(numberOfLayer - 1), "GredientW0");
            //Debug////////////////
            int i = numberOfLayer - 2;
            while(i >= 0)
            {
                layers[i].NextLinearWeightDelta = &(layers[i + 1].WeightDelta);
                layers[i].BackPropagate(learningRate);
                //Debug////////////////
                Tools::dump(*(layers[i].NextLinearWeightDelta), "NextLinearWeightDelta_" + Tools::ToString(i), "NextLinearWeightDelta");
                Tools::dump(*(layers[i].GetW()), "W_" + Tools::ToString(i), "W");
                Tools::dump(*(layers[i].GetW0()), "W0_" + Tools::ToString(i), "W0");
                Tools::dump(*(layers[i].Input()), "input_" + Tools::ToString(i), "input");
                Tools::dump(*(layers[i].GetGredientW()), "gredientW_" + Tools::ToString(i), "GredientW");
                Tools::dump(*(layers[i].GetGredientW0()), "gredientW0_" + Tools::ToString(i), "GredientW0");
                //Debug////////////////
                i -= 1;
            }
        }
        void Update(double learningRate)
        {
            for(auto &layer: this->layers)
            {
                layer.Update(learningRate);
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
        vector<LinearLayer> layers;
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
                //predict = softmax((*output)[i]);
                double sum = 0.0;
                for (size_t j = 0; j < outputSize; j++)
                {
                    sum += (*output)[i][j];             
                }
                for(size_t j = 0; j < outputSize; j++)
                {
                    if ((*labels)[i] == j)
                    {
                        deltas[i][j] = ((*output)[i][j] / sum) - 1.0;
                    }
                    else
                    {
                        deltas[i][j] = ((*output)[i][j] / sum) - 0.0;
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
