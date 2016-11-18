#ifndef _LANN_TYPES_H_
#define _LANN_TYPES_H_

#include <vector>
using std::vector;
typedef vector< vector<double> > Matrix;
typedef vector< vector<int> > MatrixInt;
typedef vector<double> Vector;
typedef vector<int> VectorInt;

typedef double (*DistanceFunc)(Vector&, Vector&);
typedef int LabelType;

#endif
