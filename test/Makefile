testM: testMuldimen.cpp
	g++ testMuldimen.cpp -o testMul -std=c++11 -g

test: test_tools.cpp
	g++ test_tools.cpp -o test_tools -std=c++11 -g

fptest: test_forwPropagate.cpp ../LinearLayer.cpp ../lznn_types.h 
	g++ test_forwPropagate.cpp -o test_forwPropagate -std=c++11 -g

bptest: test_backPropagate.cpp ../LinearLayer.cpp ../MLP.cpp ../lznn_types.h 
	g++ test_backPropagate.cpp -o test_backPropagate -std=c++11 -g

.PHONY:clean
clean:
	rm test_tools testMul dumpMatrix.txt dumpVector.txt test_backPropagate test_forwPropagate
