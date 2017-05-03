#ifndef _LZNN_CNN_
#define _LZNN_CNN_

#include "lznn_types.h"
class ConvolutionLayer
{
    public:
        CNN(){}
        void ForwPropagate()
        {

        }
        void BackPropagate(double learningRate)
        {

        }
    private:
        Matrix *filter;
        Vector *bias

        Matrix *delta;
};

#endif