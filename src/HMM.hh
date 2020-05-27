#ifndef HMM_hh
#define HMM_hh

#include <iostream>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include "../lib/Matrix/Matrix.hh"
#include "../lib/Numpy2vec/nptostdvect.hh"

using namespace std;
using vector_d = vector<double>;
using matrix_d = vector<vector_d>;
using tensor_d = vector<matrix_d>;
using vector_int = vector<int>;
using matrix_int = vector<vector_int>;
using tensor_int = vector<matrix_int>;

class HMM {
    public:
        // Constructors
        HMM(int s);
        HMM(int s, const matrix_d& mat, const matrix_d& params);

        // Functions to get HMM parameters
        int get_nstates();
        matrix_d get_params();
        matrix_d get_trans_mat();
        vector_d get_initial_dist();
        int get_decode_iter_print();
        bool get_print_decode();
        bool get_pdf_constant();
        bool get_log_em();

        // Functions to manually change parameters
        void set_nstates(int s);
        void set_params(const matrix_d& mat);
        void set_trans_mat(const matrix_d &mat);
        void set_initial_matrix();
        void set_initial_random_matrix();
        void set_initial_dist(const vector_d &v);
        void set_initial_means(const vector_d& means);
        void set_initial_vars(const vector_d& means);
        // Print related parameters
        void set_decode_iter_print(int iter);
        void set_print_decode(bool p);
        void switch_log_em();
        // Switch to true in order to change distribution parameters during EM
        void switch_pdf_constant();

        // Returns likelihood of observation
        double likelihood(const vector_d& signal);

        // Different ways of fitting the parameters
        void fit_model_params(const vector_d &signal);
        void fit_model_params_fast(const vector_d &signal);
        void fit_model_params_from_truth(const vector_d &signal, const vector_int &v_states);
        void fit_model_params(const vector_int &v_states);
        void fit_model_params(const matrix_d &probs);

        // Returns most likely sequence of states
        vector_int decode(const vector_d &signal, bool only_seq = true);

        static vector_int decode_seq(const int states, const matrix_d &matrix,
                            const vector_d &initial_dist, const matrix_d &params,
                            const matrix_d &distr, const bool prints = false,
                            const int iter_print = 10000);

        static vector_int decode_prob(const int states, const matrix_d &matrix,
                            const vector_d &initial_dist, const matrix_d &params,
                            const matrix_d &distr, const bool prints = false,
                            const int iter_print = 10000);

        void EM_maximization(const vector_d &signal,
                             const vector_int &states = vector_int(0),
                             double tol = 0.01, int maxits = 30, int seed = 0,
                             int iter_print = 1);
        

    private : 
        int nstates;
        matrix_d trans_matrix; // mat[i,j] = P(x_t+1 = j | x_t = i)
        matrix_d parameters;
        vector_d initial_dist;
        int decode_iter_print = 10000;
        bool print_decode = false;
        bool pdf_constant = true;
        bool log_em = true;
        bool init_params = true;
        bool init_matrix = true;

        static double gaussian_pdf(double x, double mu = 0, double var = 1);

        // Matrix T x N: distribution of T observations on the N classes
        matrix_d distribution(const vector_d& signal);
        vector_d distribution(double observation);

        // Fits parameters according to gaussian's max likeligood
        void gaussian_params(const vector_d& signal, const vector_int& v_states);
        void gaussian_params(const vector_d& signal, const matrix_d& gammes);
        // Fits trans matrix from different inputs
        void fit_matrix(const tensor_d &epsilons, const matrix_d &gammas);
        void fit_matrix_fast(const matrix_d &forw, const matrix_d &back,
                        const matrix_d &distrib, const matrix_d &gammas);
        void fit_matrix(const vector_int &states);
        void fit_matrix(const matrix_d &probs);
        static matrix_d path_probs(const matrix_d &matrix, 
                                        const vector_d &curr_prob,
                                        const vector_d &distr);
        // Function needed to obtain probabilities needed for EM parameter estimation                      
        matrix_d forward(const vector_d &signal, const matrix_d &distrib);
        matrix_d forward_fast(const vector_d &signal, const matrix_d &distr);
        matrix_d backward(const vector_d &signal, const matrix_d &distrib);
        matrix_d backward_fast(const vector_d &signal, const matrix_d &distrib);
        matrix_d gammation(const matrix_d &forw, const matrix_d &back);
        void gammation_fast(const matrix_d &forw, const matrix_d &back,
                                matrix_d &gammas);
        // Not used
        void epsilation(const matrix_d &forw, const matrix_d &back,
                        const matrix_d &distrib, tensor_d &epsilons);
        void epsilation(const vector_d &forw, const vector_d &back,
                        const vector_d &distrib, matrix_d &epsilons);

        void EM_maximization_vit(const vector_d &signal, double tol=0.01,
                                  int maxits = 30, int iter_print = 1);
        // Falta triar pdf i params
        

};



#endif
