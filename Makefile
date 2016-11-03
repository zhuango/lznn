test: forwPropagate_Test.cpp LinearLayer.hpp lznn_types.h 
	g++ forwPropagate_Test.cpp -o test -std=c++11
testM: testMuldimen.cpp
	g++ testMuldimen.cpp -o testMul -std=c++11

.PHONY:clean
clean:
	rm test testMul
