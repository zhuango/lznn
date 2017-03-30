#ifndef _LZNN_CNN_
#define _LZNN_CNN_

#include "lznn_types.h"
class ConvolutionLayer
{
    public:
        CNN(){}
    private:
        Matrix *filter;
        Matrix *delta;
};

#endif