/*
#ifndef matrix_d_hh
#define matrix_d_hh


// Class matrix:
class matrix_d {

    public:
        // Constructors
        matrix_d ();
        matrix_d (int N, const vector_d& row);
        
        // Operators
        matrix_d operator+(const double &right) const;
        matrix_d operator+(const matrix_d &right) const;
        matrix_d& operator+=(const matrix_d &right);

        matrix_d operator-(const double &right) const;
        matrix_d operator-(const matrix_d &right) const;
        matrix_d& operator-=(const matrix_d &right);
        
        matrix_d operator*(const double &right) const;
        matrix_d operator*(const matrix_d &right) const;
        matrix_d& operator*=(const matrix_d &right);

        matrix_d operator/(const double &right) const;
        matrix_d operator/(const matrix_d &right) const;
        matrix_d& operator/=(const matrix_d &right);

        matrix_d operator&(const matrix_d &right) const;
        vector_d operator&(const vector_d &right) const;

        vector_d operator[](const int& i);
        int size();
    private:
        vector<vector<double>> mat;
};
#endif
*/
#ifndef Matrix_h
#define Matrix_h

#include <iostream>
#include <vector>
#include <iterator>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;
using vector_d = vector<double>;
using matrix_d = vector<vector_d>;
using vector_int = vector<int>;
using matrix_int = vector<vector_int>;

matrix_d transpose(const matrix_d &matrix);
matrix_int transpose(const matrix_int &matrix);

pair<int, double> argmax(const vector_d &v);
pair<int, double> argmin(const vector_d &v);
pair<vector_int, vector_d> argmax(const matrix_d &matrix, int axis = 0);
pair<vector_int, vector_d> argmin(const matrix_d &matrix, int axis = 0);

double sum(const vector_d &v);
double sum(const matrix_d &v);
vector_d sum(const matrix_d &m, int axis);

double norm(const vector_d &v, bool do_sqrt = true);
double norm(const matrix_d &m, bool do_sqrt = true);
vector_d norm(const matrix_d &m, int axis, bool do_sqrt = true);

double dot(const vector_d &v1, const vector_d &v2);
/*
vector_d& operator+=(const double &right);
matrix_d& operator+=(const double &right);
vector_d& operator+=(const vector_d &right);
matrix_d& operator+=(const matrix_d &right);
vector_d &operator-=(const double &right);
matrix_d &operator-=(const double &right);
vector_d &operator-=(const vector_d &right);
matrix_d &operator-=(const matrix_d &right);
vector_d &operator*=(const double &right);
matrix_d &operator*=(const double &right);
vector_d &operator*=(const vector_d &right);
matrix_d &operator*=(const matrix_d &right);
vector_d &operator/=(const double &right);
matrix_d &operator/=(const double &right);
vector_d &operator/=(const vector_d &right);
matrix_d &operator/=(const matrix_d &right);
*/

vector_d operator+(const vector_d &left, const double &right);
vector_d operator+(const double &left, const vector_d &right);
matrix_d operator+(const matrix_d &left, const double &right);
matrix_d operator+(const double &left, const matrix_d &right);
vector_d operator+(const vector_d &left, const vector_d &right);
matrix_d operator+(const matrix_d &left, const matrix_d &right);

vector_d operator-(const vector_d &left, const double &right);
vector_d operator-(const double &left, const vector_d &right);
matrix_d operator-(const matrix_d &left, const double &right);
matrix_d operator-(const double &left, const matrix_d &right);
vector_d operator-(const vector_d &left, const vector_d &right);
matrix_d operator-(const matrix_d &left, const matrix_d &right);

vector_d operator*(const vector_d &left, const double &right);
vector_d operator*(const double &left, const vector_d &right);
matrix_d operator*(const matrix_d &left, const double &right);
matrix_d operator*(const double &left, const matrix_d &right);
vector_d operator*(const vector_d &left, const vector_d &right);
matrix_d operator*(const matrix_d &left, const matrix_d &right);

vector_d operator/(const vector_d &left, const double &right);
vector_d operator/(const double &left, const vector_d &right);
matrix_d operator/(const matrix_d &left, const double &right);
matrix_d operator/(const double &left, const matrix_d &right);
vector_d operator/(const vector_d &left, const vector_d &right);
matrix_d operator/(const matrix_d &left, const matrix_d &right);

matrix_d dyadic_product(const vector_d &v1, const vector_d &v2);

void print(const vector_int v, bool first = true);
void print(const vector_d v, bool first = true);
void print(const matrix_int mat, bool first= true);
void print(const matrix_d mat, bool first = true);

vector_int concat(const vector_int& v1, const vector_int& v2);
vector_d concat(const vector_d& v1, const vector_d& v2);
matrix_d concat(const matrix_d& m1, const matrix_d& m2, int axis = 0);
matrix_int concat(const matrix_int& m1, const matrix_int& m2, int axis = 0);

matrix_d operator &(const matrix_d &left, const matrix_d &right);
matrix_int operator &(const matrix_int &left, const matrix_int &right);
vector_d operator&(const matrix_d &left, const vector_d &right);
vector_d operator&(const vector_d &left, const matrix_d &right);

#endif