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
CXXFLAGS= -w -O2 -std=c++11 -fPIC -I$(PYDIRECTORY) -I$(PYDIRECTORY2) -I$(PYDIRECTORYTONI) -I$(NPDIRECTORY) -I$(NPDIRECTORY2) -I$(NPDIRECTORY3) -I$(NPDIRECTORYTONI) -I$(ALGUNALTRE)
DEPS=./lib/Matrix/Matrix.hh ./lib/Numpy2vec/nptostdvect.hh
OBJ=./lib/Matrix/Matrix.o ./lib/Numpy2vec/nptostdvect.o ./src/HMM.o ./src/HMM_wrap.o ./src/HiddenHMM.o ./src/HiddenHMM_comp.o 

%.o:%.cc $(DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

./dist/_HMM.so: $(OBJ) 
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

./src/HMM_wrap.cc: ./src/HMM.i ./src/HMM.o
	swig -c++ -python ./src/HMM.i
	mv ./src/HMM_wrap.cxx ./src/HMM_wrap.cc

clean:
	rm ./lib/*/*.o ./dist/*.so ./src/HMM_wrap.cc ./src/HMM.py
