#ifndef _KNEIGHBORSCLASSIFIER_CPP_
#define _KNEIGHBORSCLASSIFIER_CPP_

#include "lznn_types.h"
#include "lznn_tools.cpp"

#include <utility>
#include <queue>
using namespace std;

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

typedef priority_queue< pair<int, double>, vector< pair<int, double> >, KNeighborsClassifierComparison > VotePriorityQueue; 

class KNeighborsClassifier
{
    public:
        KNeighborsClassifier(size_t nNeighbors, size_t nLabel)
        :
        nNeighbors (nNeighbors),
        nLabel     (nLabel)
        {}

        void Fit(Matrix *inputs, VectorInt *labels, DistanceFunc distance = Tools::EuclideanDistance)
        {
            this->trainInputs = inputs;
            this->trainLabels = labels;
            this->distance    = distance;
        }

        void Predict(Matrix &inputs, VectorInt &labels)
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                LabelType result = predictOne(inputs[i]);
                labels.push_back(result);
            }
        }
        
    private:

        size_t       nNeighbors;
        size_t       nLabel;
        Matrix       *trainInputs;
        VectorInt    *trainLabels;
        DistanceFunc distance;

        LabelType predictOne(Vector &input)
        {
            double curDistance = 0.0;
            VotePriorityQueue maxHeap;
            for (size_t j = 0; j < trainInputs->size(); j++)
            {
                curDistance = distance(input, (*trainInputs)[j]);
                if     (maxHeap.size() < nNeighbors)
                {
                    maxHeap.push(pair<int, double>(j, curDistance));
                }
                else if(maxHeap.top().second > curDistance)
                {
                    maxHeap.pop();
                    maxHeap.push(pair<int, double>(j, curDistance));
                }
            }
            LabelType result = majority(maxHeap);
            return result;
        }
        LabelType majority(VotePriorityQueue &maxHeap)
        {
            vector<int> votes(nLabel, 0);
            LabelType result = 0;
            while(!maxHeap.empty())
            {
                int index = maxHeap.top().first;
                LabelType label = (*trainLabels)[index];
                votes[label] += 1;
                if (votes[label] > votes[result])
                    result = label;
                maxHeap.pop();
            }
            return result;
        }
};


#endif