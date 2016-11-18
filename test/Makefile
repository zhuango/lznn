testM: testMuldimen.cpp
	g++ testMuldimen.cpp -o testMul -std=c++11 -g

test: test_tools.cpp
	g++ test_tools.cpp -o test_tools -std=c++11 -g

fptest: test_forwPropagate.cpp ../src/lznn_PerceptronLayer.cpp ../src/lznn_types.h 
	g++ test_forwPropagate.cpp -o test_forwPropagate -std=c++11 -g

bptest: test_backPropagate.cpp ../src/lznn_PerceptronLayer.cpp ../src/lznn_MLP.cpp ../src/lznn_types.h 
	g++ test_backPropagate.cpp -o test_backPropagate -std=c++11 -g

bpRtest: test_backPropagateRegularizedPerceptronLayer.cpp ../src/lznn_PerceptronLayer.cpp ../src/lznn_RegularizedPerceptronLayer.cpp ../src/lznn_MLP.cpp ../src/lznn_types.h 
	g++ test_backPropagateRegularizedPerceptronLayer.cpp -o test_backPropagateRegularizedPerceptronLayer -std=c++11 -g

test_performance: test_performance.cpp ../src/lznn_PerceptronLayer.cpp ../src/lznn_MLP.cpp ../src/lznn_types.h 
	g++ test_performance.cpp -o test_performance -std=c++11 -g

knnTest: test_KNeighborsClassifier.cpp ../src/lznn_KNeighborsClassifier.cpp
	g++ test_KNeighborsClassifier.cpp -o test_KNeighborsClassifier -std=c++11 -g

.PHONY:clean
clean:
	rm test_tools testMul dumpMatrix.txt dumpVector.txt test_backPropagate test_forwPropagate test_KNeighborsClassifier test_performance test_backPropagateRegularizedPerceptronLayer
