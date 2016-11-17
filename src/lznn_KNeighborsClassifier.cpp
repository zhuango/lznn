#ifndef _KNEIGHBORSCLASSIFIER_CPP_
#define _KNEIGHBORSCLASSIFIER_CPP_
#include "lznn_types.h"
#include <queue>
#include <map>

class KNeighborsClassifierComparison
{
  bool reverse;
public:
  KNeighborsClassifierComparison(const bool& revparam=false)
    {reverse=revparam;}
  bool operator() (const pair<int, double>& lhs, const pair<int, double>&rhs) const
  {
    if (reverse) return (lhs.second > rhs.second);
    else return (lhs.second < rhs.second);
  }
};

class KNeighborsClassifier
{
    public:
        KNeighborsClassifier(size_t nNeighbors, size_t nLabel)
        :
        nNeighbors (nNeighbors),
        nLabel     (nLabel)
        {}

        void Fit(Matrix *inputs, VectorInt *labels, DistanceFunc distance = euclideanDistance)
        {
            this->trainInputs = inputs;
            this->trainLabels = labels;
            this->distance    = distance;
        }

        void Predict(Matrix &inputs, VectorInt &labels)
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                predictOne(inputs[i], labels[i])
            }
        }
        void predictOne(Vector &input, LabelType &label)
        {
            double curDistance = 0.0;
            priority_queue< pair<int, double>, vector< pair<int, double> >, KNeighborsClassifierComparison > maxHeap;

            for (size_t j = 0; j < trainInputs->size(); j++)
            {
                curDistance = distance(input, trainInputs[j])
                if     (maxHeap.size() < nNeighbors)
                {
                    maxHeap.push(pair<int, double>(j, curDistance))
                }
                else if(maxHeap.top().second > curDistance)
                {
                    maxHeap.pop();
                    maxHeap.push(pair<int, double>(j, curDistance))
                }
            }
            vector<int> votes(nLabel, 0);
            LabelType result = 0;
            while(!maxHeap.empty())
            {
                int index = maxHeap.top().first;
                votes[index] += 1;
                if votes[index] > votes[result]:
                    result = index;
                maxHeap.pop();
            }
            label = result;
        }
    private:
        typedef double (*DistanceFunc)(Vector, Vector);
        trpedef int LabelType;

        size_t       nNeighbors;
        size_t       nLabel;
        Matrix       *trainInputs;
        Vector       *trainLabels;
        DistanceFunc distance;

        double euclideanDistance(Vector a, Vector b)
        {
            if (a.size() != b.size()) {throw;}

            double result = 0;
            for (int i = 0, i < a.size(); i++)
            {
                result += pow((a[i] - b[i]), 2.0);
            }
            result = sqrt(result);
            return result;
        }
};


#endif