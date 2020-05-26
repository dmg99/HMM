#include "Matrix.hh"
#include <iostream>

using namespace std;

int main(){
    matrix_d toni(4, vector_d(3, 1));
    matrix_d dani(4, vector_d(3, 5));

    print(toni/dani);      
}
