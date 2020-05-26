#include "Matrix.hh"

#include <iostream>
#include <vector>
#include <iterator>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <float.h>
#include <stdlib.h>

using namespace std;
using vector_d = vector<double>;
using matrix_d = vector<vector_d>;
using vector_int = vector<int>;
using matrix_int = vector<vector_int>;
using tensor_d = vector<matrix_d>;
// Transpose functions
matrix_d transpose(const matrix_d &matrix)
{
    int n = matrix.size();
    int m = matrix[0].size();

    matrix_d result(m, vector_d(n));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
            result[j][i] = matrix[i][j];
    }

    return result;
}

matrix_int transpose(const matrix_int &matrix)
{
    int n = matrix.size();
    int m = matrix[0].size();

    matrix_int result(m, vector_int(n));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
            result[j][i] = matrix[i][j];
    }

    return result;
}

// Argmax/Argmin functions
pair<int, double> argmax(const vector_d &v)
{
    pair<int, double> ret;
    auto maxptr = max_element(v.begin(), v.end());
    ret.first = distance(v.begin(), maxptr);
    ret.second = *maxptr;
    return ret;
}

pair<int, double> argmin(const vector_d &v)
{
    pair<int, double> ret;
    auto maxptr = min_element(v.begin(), v.end());
    ret.first = distance(v.begin(), maxptr);
    ret.second = *maxptr;
    return ret;
}

pair<vector_int, vector_d> argmax(const matrix_d &matrix, int axis)
{
    assert(axis >= 0 and axis <= 1);

    matrix_d matrix_it = ((axis == 0) ? matrix : transpose(matrix));
    int n = matrix_it.size();

    pair<vector_int, vector_d> result;
    result.first = vector_int(n);
    result.second = vector_d(n);

    for (int i = 0; i < n; ++i)
    {
        pair<int, double> curr_result = argmax(matrix_it[i]);
        result.first[i] = curr_result.first;
        result.second[i] = curr_result.second;
    }

    return result;
}

pair<vector_int, vector_d> argmin(const matrix_d &matrix, int axis)
{
    assert(axis >= 0 and axis <= 1);

    matrix_d matrix_it = ((axis == 0) ? matrix : transpose(matrix));
    int n = matrix_it.size();

    pair<vector_int, vector_d> result;
    result.first = vector_int(n);
    result.second = vector_d(n);

    for (int i = 0; i < n; ++i)
    {
        pair<int, double> curr_result = argmin(matrix_it[i]);
        result.first[i] = curr_result.first;
        result.second[i] = curr_result.second;
    }

    return result;
}

pair<int, int> mat_argmax(const matrix_d& matrix){
    double best_max = -DBL_MAX;
    pair<int, int> result;
    int n = matrix.size();
    int m = matrix[0].size();
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            if (matrix[i][j] > best_max){
                best_max = matrix[i][j];
                result.first = i;
                result.second = j;
            }
        }
    }
    return result;
}
// Sum functions
double sum(const vector_d& v)
{
    double s = 0;
    for(const double x: v){
        s += x;
    }
    return s;
}

double sum(const matrix_d &m)
{
    double s = 0;
    for (const vector_d x : m)
    {
        s += sum(x);
    }
    return s;
}

vector_d sum(const matrix_d& m, int axis){
    matrix_d mat = m;
    if (axis==1) {
        mat = transpose(m);
    }

    int n = mat.size();
    vector_d v(n);
    for (int i = 0; i < n; i++)
    {
        v[i] = sum(mat[i]);
    }
    
    return v;
}

// Norm functions
double norm(const vector_d &v, bool do_sqrt){
    double s = 0;
    for (const double x : v)
        s += x*x;
    if (do_sqrt) 
        s = sqrt(s);
    return s;
}

double norm(const matrix_d &m, bool do_sqrt){
    double s = 0;
    for (const vector_d x : m)
        s += norm(x, false);
    
    if (do_sqrt)
        s = sqrt(s);
    return s;
}

double norm(const tensor_d &m, bool do_sqrt){
    int n = m.size();
    double s = 0;
    for (int i = 0; i < n; ++i){
        s += norm(m[i], false);
    }
    if (do_sqrt)
        s = sqrt(s);
    return s;
}
vector_d norm(const matrix_d &m, int axis, bool do_sqrt)
{
    matrix_d mat = (axis==0) ? m : transpose(m);

    int n = mat.size();
    vector_d v(n);
    for (int i = 0; i < n; i++)
        v[i] = norm(mat[i], do_sqrt);

    return v;
}

double dot(const vector_d &v1, const vector_d &v2){
    int N = v1.size(),
        N2 = v2.size();

    assert(N==N2);

    double sum = 0;
    for (int n = 0; n < N; n++)
    {
        sum += v1[n] * v2[n];
    }

    return sum;
}

/*
// Operators +=
vector_d& operator+=(const double &right)
{
    int N = *this.size();

    for (int i = 0; i < N; ++i)
    {
        *this[i] += right;
    }

    return *this;
}

matrix_d& operator+=(const double &right)
{
    assert(*this.size()>0);

    int N = *this.size(),
        M = *this[0].size();

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; j++)
        {
            *this[i] += right;
        }
    }

    return *this;
}

vector_d& operator+=(const vector_d &right)
{
    int N = *this.size(),
        N2 = right.size();

    assert(N == N2);

    for (int i = 0; i < N; ++i)
    {
        *this[i] += right[i];
    }

    return *this;
}

matrix_d& operator+=(const matrix_d &right)
{
    int N = left.size(),
        N2 = left[0].size(),
        M = right.size(),
        M2 = right[0].size();

    assert(N == N2 and M == M2);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            *this[i][j] += right[i][j];
        }
    }

    return *this;
}

// Operators -=
vector_d &operator-=(const double &right)
{
    int N = *this.size();

    for (int i = 0; i < N; ++i)
    {
        *this[i] -= right;
    }

    return *this;
}

matrix_d &operator-=(const double &right)
{
    assert(*this.size() > 0);

    int N = *this.size(),
        M = *this[0].size();

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; j++)
        {
            *this[i] -= right;
        }
    }

    return *this;
}

vector_d &operator-=(const vector_d &right)
{
    int N = *this.size(),
        N2 = right.size();

    assert(N == N2);

    for (int i = 0; i < N; ++i)
    {
        *this[i] -= right[i];
    }

    return *this;
}

matrix_d &operator-=(const matrix_d &right)
{
    int N = left.size(),
        N2 = left[0].size(),
        M = right.size(),
        M2 = right[0].size();

    assert(N == N2 and M == M2);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            *this[i][j] -= right[i][j];
        }
    }

    return *this;
}
// Operators *=
vector_d &operator*=(const double &right)
{
    int N = *this.size();

    for (int i = 0; i < N; ++i)
    {
        *this[i] *= right;
    }

    return *this;
}

matrix_d &operator*=(const double &right)
{
    assert(*this.size() > 0);

    int N = *this.size(),
        M = *this[0].size();

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; j++)
        {
            *this[i] *= right;
        }
    }

    return *this;
}

vector_d &operator*=(const vector_d &right)
{
    int N = *this.size(),
        N2 = right.size();

    assert(N == N2);

    for (int i = 0; i < N; ++i)
    {
        *this[i] *= right[i];
    }

    return *this;
}

matrix_d &operator*=(const matrix_d &right)
{
    int N = left.size(),
        N2 = left[0].size(),
        M = right.size(),
        M2 = right[0].size();

    assert(N == N2 and M == M2);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            *this[i][j] *= right[i][j];
        }
    }

    return *this;
}

// Operators /=
vector_d &operator/=(const double &right)
{
    int N = *this.size();

    for (int i = 0; i < N; ++i)
    {
        *this[i] /= right;
    }

    return *this;
}

matrix_d &operator/=(const double &right)
{
    assert(*this.size() > 0);

    int N = *this.size(),
        M = *this[0].size();

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; j++)
        {
            *this[i] /= right;
        }
    }

    return *this;
}

vector_d &operator/=(const vector_d &right)
{
    int N = *this.size(),
        N2 = right.size();

    assert(N == N2);

    for (int i = 0; i < N; ++i)
    {
        *this[i] /= right[i];
    }

    return *this;
}

matrix_d &operator/=(const matrix_d &right)
{
    int N = left.size(),
        N2 = left[0].size(),
        M = right.size(),
        M2 = right[0].size();

    assert(N == N2 and M == M2);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            *this[i][j] /= right[i][j];
        }
    }

    return *this;
}
*/

// Operators +
vector_d operator+(const vector_d &left, const double &right)
{
    int N = left.size();

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left[i] + right;
    }

    return result;
}

vector_d operator+(const double &left, const vector_d &right)
{
    int N = right.size();

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left + right[i];
    }

    return result;
}

matrix_d operator+(const matrix_d &left, const double &right)
{
    int N = left.size();

    matrix_d result(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            result[i] = left[i] + right;
        }
    }

    return result;
}

matrix_d operator+(const double &left, const matrix_d &right)
{
    int N = right.size();

    matrix_d result(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            result[i] = left + right[i];
        }
    }

    return result;
}

vector_d operator+(const vector_d &left, const vector_d &right)
{
    int N = left.size(),
        N2 = right.size();

    assert(N == N2);

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left[i] + right[i];
    }

    return result;
}

matrix_d operator+(const matrix_d &left, const matrix_d &right)
{
    int N = left.size(),
        N2 = left[0].size(),
        M = right.size(),
        M2 = right[0].size();

    assert(N == M and M2 == N2);

    matrix_d result(N, vector_d(N2));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N2; ++j)
        {
            result[i][j] = left[i][j] + right[i][j];
        }
    }

    return result;
}

tensor_d operator+(const tensor_d & left, const tensor_d& right){
    int n1 = left.size(),
        n2 = right.size(),
        m1 = left[0].size(),
        m2 = right[0].size(),
        k1 = left[0][0].size(),
        k2 = right[0][0].size();
        
    assert(n1 == n2 and m1 == m2 and k1 == k2);
    tensor_d result(n1, matrix_d(m1, vector_d(k1, 0)));
    for (int i = 0; i < n1; ++i){
        result[i] = left[i]+right[i];
    }
    return result;
}
// Operators -
vector_d operator-(const vector_d &left, const double &right)
{
    int N = left.size();

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left[i] - right;
    }

    return result;
}

vector_d operator-(const double &left, const vector_d &right)
{
    int N = right.size();

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left - right[i];
    }

    return result;
}

matrix_d operator-(const matrix_d &left, const double &right)
{
    int N = left.size();

    matrix_d result(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            result[i] = left[i] - right;
        }
    }

    return result;
}

matrix_d operator-(const double &left, const matrix_d &right)
{
    int N = right.size();

    matrix_d result(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            result[i] = left - right[i];
        }
    }

    return result;
}

vector_d operator-(const vector_d &left, const vector_d &right)
{
    int N = left.size(),
        N2 = right.size();

    assert(N == N2);

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left[i] - right[i];
    }

    return result;
}

matrix_d operator-(const matrix_d &left, const matrix_d &right)
{
    int N = left.size(),
        N2 = left[0].size(),
        M = right.size(),
        M2 = right[0].size();

    assert(N == M and N2 == M2);

    matrix_d result(N, vector_d(N2));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N2; ++j)
        {
            result[i][j] = left[i][j] - right[i][j];
        }
    }

    return result;
}

tensor_d operator-(const tensor_d & left, const tensor_d& right){
    int n1 = left.size(),
        n2 = right.size(),
        m1 = left[0].size(),
        m2 = right[0].size(),
        k1 = left[0][0].size(),
        k2 = right[0][0].size();
        
    assert(n1 == n2 and m1 == m2 and k1 == k2);
    tensor_d result(n1, matrix_d(m1, vector_d(k1, 0)));
    for (int i = 0; i < n1; ++i){
        result[i] = left[i]-right[i];
    }
    return result;
}

// Operators *
vector_d operator*(const vector_d &left, const double &right)
{
    int N = left.size();

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left[i] * right;
    }

    return result;
}

vector_d operator*(const double &left, const vector_d &right)
{
    int N = right.size();

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left * right[i];
    }

    return result;
}

matrix_d operator*(const matrix_d &left, const double &right)
{
    int N = left.size();

    matrix_d result(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            result[i] = left[i] * right;
        }
    }

    return result;
}

matrix_d operator*(const double &left, const matrix_d &right)
{
    int N = right.size();

    matrix_d result(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            result[i] = left * right[i];
        }
    }

    return result;
}

vector_d operator*(const vector_d &left, const vector_d &right)
{
    int N = left.size(),
        N2 = right.size();

    assert(N == N2);

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left[i] * right[i];
    }

    return result;
}

matrix_d operator*(const matrix_d &left, const matrix_d &right)
{
    int N = left.size(),
        N2 = left[0].size(),
        M = right.size(),
        M2 = right[0].size();

    assert(N == M and M2 == N2);

    matrix_d result(N, vector_d(N2));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N2; ++j)
        {
            result[i][j] = left[i][j] * right[i][j];
        }
    }

    return result;
}

// Operators /
vector_d operator/(const vector_d &left, const double &right)
{
    int N = left.size();

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left[i] / right;
    }

    return result;
}

vector_d operator/(const double &left, const vector_d &right)
{
    int N = right.size();

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left / right[i];
    }

    return result;
}

matrix_d operator/(const matrix_d &left, const double &right)
{
    int N = left.size();

    matrix_d result(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            result[i] = left[i] / right;
        }
    }

    return result;
}

matrix_d operator/(const double &left, const matrix_d &right)
{
    int N = right.size();

    matrix_d result(N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; j++)
        {
            result[i] = left / right[i];
        }
    }

    return result;
}

vector_d operator/(const vector_d &left, const vector_d &right)
{
    int N = left.size(),
        N2 = right.size();

    assert(N == N2);

    vector_d result(N);
    for (int i = 0; i < N; ++i)
    {
        result[i] = left[i] / right[i];
    }

    return result;
}

matrix_d operator/(const matrix_d &left, const matrix_d &right)
{
    int N = left.size(),
        N2 = left[0].size(),
        M = right.size(),
        M2 = right[0].size();

    assert(N == M and N2 == M2);

    matrix_d result(N, vector_d(N2));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N2; ++j)
        {
            result[i][j] = left[i][j] / right[i][j];
        }
    }

    return result;
}

matrix_d dyadic_product(const vector_d &v1, const vector_d &v2)
{
    int N = v1.size(),
        M = v2.size();

    matrix_d matrix(N, vector_d(M));
    for(int i=0; i<N; ++i){
        for(int j=0; j<M; ++j){
            matrix[i][j] = v1[i]*v2[j];
        }
    }

    return matrix;
}

// Print functions
void print(const vector_int v, bool first)
{
    int n = v.size();
    if (first)
        cout << "Vector:" << endl;
    if (n == 0)
        return;
    cout << "   " << v[0];
    for (int i = 1; i < n; ++i)
        cout << ", " << v[i];
    cout << ";" << endl;
}

void print(const vector_d v, bool first)
{
    int n = v.size();
    if (first)
        cout << "Vector:" << endl;
    if (n == 0)
        return;
    cout << "   " << v[0];
    for (int i = 1; i < n; ++i)
        cout << ", " << v[i];
    cout << ";" << endl;
}

void print(const matrix_int mat, bool first)
{
    int n = mat.size(),
        m = mat[0].size();
    if (first)
        cout << "Matrix:" << endl;
    if (n == 0 or m==0)
        return;
    for (int i = 0; i < n; ++i)
        print(mat[i], false);
}

void print(const matrix_d mat, bool first)
{
    int n = mat.size(),
        m = mat[0].size();
    if (first)
        cout << "Matrix:" << endl;
    if (n == 0 or m == 0)
        return;
    for (int i = 0; i < n; ++i)
        print(mat[i], false);
}

vector_int concat(const vector_int& v1, const vector_int& v2){
    vector_int result = v1;
    result.insert(result.end(), v2.begin(), v2.end());
    return(result);
}

vector_d concat(const vector_d& v1, const vector_d& v2){
    vector_d result = v1;
    result.insert(result.end(), v2.begin(), v2.end());
    return(result);
}

matrix_d concat(const matrix_d& m1, const matrix_d& m2, int axis){
    matrix_d mat1;
    matrix_d mat2;
    if (axis == 1){
        mat1 = m1;
        mat2 = m2;
    }
    else{
        mat1 = transpose(m1);
        mat2 = transpose(m2);
    }
    int n1 = mat1[0].size();
    int n2 = mat2[0].size();
    
    assert(n1 == n2);
    mat1.insert(mat1.end(), mat2.begin(), mat2.end());
    
    if (axis == 1){
        return(mat1);
    }
    else{
        return (transpose(mat1));
    }
}

matrix_int concat(const matrix_int& m1, const matrix_int& m2, int axis){
    matrix_int mat1;
    matrix_int mat2;
    if (axis == 1){
        mat1 = m1;
        mat2 = m2;
    }
    else{
        mat1 = transpose(m1);
        mat2 = transpose(m2);
    }
    int n1 = mat1[0].size();
    int n2 = mat2[0].size();
    
    assert(n1 == n2);
    mat1.insert(mat1.end(), mat2.begin(), mat2.end());
    
    if (axis == 1){
        return(mat1);
    }
    else{
        return (transpose(mat1));
    }
}


matrix_d operator &(const matrix_d &left, const matrix_d &right){
	int k1 = left.size();
	int k2 = right[0].size();
	int k3 = right.size();
    int k4 = left[0].size();
    assert(k3 == k4);
	matrix_d ans(k1,vector_d(k2));
	
	for (int i = 0; i < k1; ++i){
		for (int j = 0; j < k2; ++j){
			ans[i][j]=0;
			for (int z = 0; z < k3; ++z) ans[i][j] += left[i][z]*right[z][j]; 		
		}
	}
	return ans;
}

matrix_int operator &(const matrix_int &left, const matrix_int &right){
	int k1 = left.size();
	int k2 = right[0].size();
	int k3 = right.size();
    int k4 = left[0].size();
    assert(k3 == k4);
	matrix_int ans(k1,vector_int(k2));
	
	for (int i = 0; i < k1; ++i){
		for (int j = 0; j < k2; ++j){
			ans[i][j]=0;
			for (int z = 0; z < k3; ++z) ans[i][j] += left[i][z]*right[z][j]; 		
		}
	}
	return ans;
}

vector_d operator&(const matrix_d &left, const vector_d &right){
    int n = right.size();
    int m1 = left.size();
    int m2 = left[0].size();
    assert(m2 == n);

    vector_d result(m1, 0);
    for (int i = 0; i < m1; ++i){
        for (int j = 0; j < n; ++j){
            result[i] += right[j]*left[i][j];
        }
    }

    return result;
}

vector_d operator&(const vector_d &left, const matrix_d &right){
    int n = left.size();
    int m1 = right.size();
    int m2 = right[0].size();
    assert(m1 == n);

    vector_d result(m2, 0);
    for (int i = 0; i < m2; ++i){
        for (int j = 0; j < n; ++j){
            result[i] += left[j]*right[j][i];
        }
    }

    return result;
}

matrix_d random_mat(int n, int m, int max_val, int seed){
    if (seed > 0){
        srand(seed);
    }
    matrix_d res(n, vector_d(m, 0));
    for (int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            res[i][j] = rand() % max_val;
        }
    }
    return res;
}