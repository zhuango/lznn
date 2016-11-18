#include "../src/lznn_KNeighborsClassifier.cpp"
#include "common.h"

int main(void)
{
    Matrix    inputs;
    Matrix    testInputs;
    VectorInt labels;
    VectorInt testLabels;
    VectorInt predicts;

    fillData("../data/knn_X_train.txt", inputs);
    fillData("../data/knn_Y_train.txt", labels, true);
    fillData("../data/knn_X_test.txt", testInputs);
    fillData("../data/knn_Y_test.txt", testLabels, true);

    KNeighborsClassifier classifier(3, 3);
    classifier.Fit(&inputs, &labels);
    classifier.Predict(testInputs, predicts);

    cout << "predict: " << endl;
    for (auto &item : predicts)
    {
        cout << item << " ";
    }
    cout << endl;

    cout << "gold   : " << endl;
    for (auto &item : testLabels)
    {
        cout << item << " ";
    }
    cout << endl;

    return 0;
}