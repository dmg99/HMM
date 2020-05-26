#include <arrayobject.h>
#include <iostream>
#include <vector>

#include "nptostdvect.hh"
using namespace std;

vector<double> array2vec (double* arr, int size = 0){
    vector<double> ret(arr, arr+size);
    return ret;
}


vector<double> numpy2vec (PyObject* numpy_array){
    Py_Initialize();
    _import_array();
    npy_intp vsize = PyArray_SIZE(numpy_array);
    double* init_ptr;
    init_ptr = (double *)PyArray_GETPTR1(numpy_array,0);
    return array2vec(init_ptr, vsize);
}

vector<int> array2vec_int(long* arr, int size = 0){
    vector<int> ret(arr, arr+size);
    return ret;
}


vector<int> numpy2vec_int(PyObject* numpy_array){
    Py_Initialize();
    _import_array();
    npy_intp vsize = PyArray_SIZE(numpy_array);
    long* init_ptr;
    init_ptr = (long *)PyArray_GETPTR1(numpy_array,0);
    return array2vec_int(init_ptr, vsize);
}
