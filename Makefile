test: test.cpp LinearLayer.hpp 
	g++ test.cpp -o test -std=c++11
.PHONY:clean
clean:
	rm test
