#ifndef nptostdvect_h
#define nptostdvect_h

#include <arrayobject.h>
#include <iostream>
#include <vector>

using namespace std;

vector<double> numpy2vec (PyObject* numpy_array);
vector<double> array2vec (double* arr, int size);
vector<int> array2vec_int(long* arr, int size);
vector<int> numpy2vec_int(PyObject* numpy_array);

#endif