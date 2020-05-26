CXXFLAGS = -Wall -std=c++11 -O2

all: main.exe

clean:
	rm -f main.exe *.o

main.exe: main.o Matrix.o
	$(CXX) $^ -o $@

main.cc: Matrix.hh