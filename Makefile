fptest: forwPropagate_Test.cpp LinearLayer.hpp lznn_types.h 
	g++ forwPropagate_Test.cpp -o fptest -std=c++11
bptest: backPropagate_Test.cpp LinearLayer.hpp MLP.hpp lznn_types.h 
	g++ backPropagate_Test.cpp -o bptest -std=c++11
testM: testMuldimen.cpp
	g++ testMuldimen.cpp -o testMul -std=c++11

.PHONY:clean
clean:
	rm test testMul
