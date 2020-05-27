CXX=g++
PYDIRECTORY=/usr/include/python3.6m
NPDIRECTORY=/usr/local/lib/python3.6/dist-packages/numpy/core/include/numpy/
CXXFLAGS= -w -O2 -std=c++11 -fPIC -I$(PYDIRECTORY) -I$(NPDIRECTORY)
DEPS=./lib/Matrix/Matrix.hh ./lib/Numpy2vec/nptostdvect.hh
OBJ=./lib/Matrix/Matrix.o ./lib/Numpy2vec/nptostdvect.o ./src/HMM.o ./src/HMM_wrap.o ./src/HiddenHMM.o ./src/AdaptiveHHMM.o 

%.o:%.cc $(DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

./dist/_HMM.so: $(OBJ) 
	$(CXX) $(CXXFLAGS) -shared $^ -o $@

./src/HMM_wrap.cc: ./src/HMM.i ./src/HMM.o
	swig -c++ -python ./src/HMM.i
	mv ./src/HMM_wrap.cxx ./src/HMM_wrap.cc
	mv ./src/HMM.py ./dist/HMM.py

clean:
	rm ./lib/*/*.o ./dist/*.so ./src/HMM_wrap.cc ./dist/HMM.py ./src/*.o
	rm -r ./dist/__pycache__
