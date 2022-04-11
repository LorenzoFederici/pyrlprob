dir = ./tests/cpp_tests/
exec = $(dir)landing1d_cpp$(shell python3-config --extension-suffix)
INC = $(shell python3 -m pybind11 --includes) $(shell python3 -m pyrlprob --includes)
LIB = 

objects = $(dir)landing1d.o

COMP?= g++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp

${exec} : ${objects}
	${COMP} -o ${exec} ${objects} $(INC) $(LIB)

$(dir)landing1d.o : $(dir)binding.cpp
	${COMP} -c $< -o $@ $(INC) $(LIB)


.PHONY : clean
clean:
	rm ${exec} ${objects}