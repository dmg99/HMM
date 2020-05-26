CXX=g++
PYDIRECTORYTONI=/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m
NPDIRECTORYTONI=/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/numpy/core/include/numpy
ALGUNALTRE=/Library/Frameworks/Python.framework/Versions/3.7/include/python3.7m
PYDIRECTORY=/usr/include/python3.6m
PYDIRECTORY2=/usr/include/python3.8
NPDIRECTORY=/usr/local/lib/python3.6/dist-packages/numpy/core/include/numpy/
NPDIRECTORY2=/usr/lib/python3.6/site-packages/numpy/core/include/numpy/
NPDIRECTORY3=/home/joobz/.local/lib/python3.8/site-packages/numpy/core/include/numpy/
LLIBRERIA = /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin
CXXFLAGS= -Wall -O2 -std=c++11 -fPIC -I$(PYDIRECTORY) -I$(PYDIRECTORY2) -I$(PYDIRECTORYTONI) -I$(NPDIRECTORY) -I$(NPDIRECTORY2) -I$(NPDIRECTORY3) -I$(NPDIRECTORYTONI) -I$(ALGUNALTRE)
DEPS=./Matrix/Matrix.hh ./Numpy2vec/nptostdvect.hh
OBJ=./Matrix/Matrix.o ./Numpy2vec/nptostdvect.o HMM.o HMM_wrap.o HiddenHMM.o HiddenHMM_comp.o 

%.o:%.cc $(DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

_HMM.so: $(OBJ) 
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

HMM_wrap.cc: HMM.i HMM.o
	swig -c++ -python HMM.i
	mv HMM_wrap.cxx HMM_wrap.cc

clean:
	rm *.o */*.o *.so HMM_wrap.cc
