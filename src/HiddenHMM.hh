#ifndef HiddenHMM_hh
#define HiddenHMM_hh

#include <iostream>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include "Matrix/Matrix.hh"
#include "Numpy2vec/nptostdvect.hh"

using namespace std;
using vector_d = vector<double>;
using matrix_d = vector<vector_d>;
using tensor_d = vector<matrix_d>;
using vector_int = vector<int>;
using matrix_int = vector<vector_int>;
using tensor_int = vector<matrix_int>;

class HiddenHMM {
    public:
        // Constructors
        HiddenHMM(int s, int h);

        // Functions to get HMM parameters
        int get_nstates();
        int get_nhidden();
        matrix_d get_params();
        
        matrix_d get_trans_mat(int h);
        tensor_d get_trans_mat();
        matrix_d get_hidden_mat();
        vector_d get_initial_dist(int h);
        matrix_d get_initial_dist();
        
        int get_decode_iter_print();
        bool get_print_decode();
        bool get_pdf_constant();
        bool get_log_em();

        // Functions to manually change parameters
        void set_nstates(int s);
        void set_nhidden(int h);

        void set_params(const matrix_d& mat);
        
        void set_trans_mat(const tensor_d& mats);
        void set_trans_mat(const matrix_d &mat, int h);
        void set_hidden_mat(const matrix_d &mat);
        
        
        void set_initial_matrix(int h, int add);
        void set_initial_random_matrix(int h);
        void set_initial_hidden();
        void set_initial_dist(const matrix_d& m);
        void set_initial_dist(const vector_d &v, int h);
        void set_initial_dist(const vector_d& hidden_dist, const vector_d& state_dist);
        
        void set_initial_means(const vector_d& means);
        void set_initial_vars(const vector_d& means);
        
        // Print related parameters
        void set_decode_iter_print(int iter);
        void set_print_decode(bool p);
        void switch_log_em();
        void switch_init();
        // Switch to true in order to change distribution parameters during EM
        void switch_pdf_constant();
        void switch_state_training();
        // Different ways of fitting the parameters
        void fit_model_params(const vector_d& signal, const matrix_d& hidden_mat,
                                const tensor_d& trans_mats);
        void fit_model_params_from_truth(const vector_d &signal, const vector_int &v_states);
        void fit_model_params_states(const vector_d& signal, const vector_int &states, 
                                const matrix_d& hidden_mat, const tensor_d& trans_mats);
                        
        // Returns most likely sequence of states
        matrix_int decode(const vector_d &signal, bool only_seq = true);
        tensor_d decode_probs(const vector_d &signal);
        static matrix_int decode_seq(const int& S, const int& H, const matrix_d& hidden_mat,
                                        const tensor_d &matrices, const matrix_d &initial_distr, const matrix_d &params,
                                        const matrix_d &distr, const bool prints,
                                        const int iter_print = 10000);
        tensor_d decode_seq_probs(const int& S, const int& H, const matrix_d& hidden_mat,
                    const tensor_d &matrices, const matrix_d &initial_distr, const matrix_d &params,
                    const matrix_d &distr, const bool prints,
                    const int iter_print);
        
        
        void EM_maximization(const vector_d &signal,
                             const vector_int &states = vector_int(0),
                             double tol = 0.01, int maxits = 30, int seed = 0,
                             int iter_print = 1);
        
    private : 
        int nstates;
        int nhidden;
        tensor_d trans_matrix; // H x S x S
        matrix_d hidden_matrix; // H x H
        matrix_d parameters;
        matrix_d initial_dist; // H x S
        int decode_iter_print = 10000;
        bool print_decode = false;
        bool pdf_constant = true;
        bool log_em = true;
        bool init_params = true;
        bool init_matrix = true;
        bool state_training = false;

        
        static double gaussian_pdf(double x, double mu = 0, double var = 1);
        
        // Matrix T x N: distribution of T observations on the N classes
        matrix_d distribution(const vector_d& signal);
        vector_d distribution(double observation);

        // Fits parameters according to gaussian's max likelihood
        void gaussian_params(const vector_d& signal, const vector_int& v_states);
        void gaussian_params(const vector_d& signal, const matrix_d& gammes);
        
        // Fits trans matrix from different inputs
        void fit_matrix(const tensor_d &forw, const tensor_d &back, 
                        const matrix_d &distrib, const matrix_d& old_hidden_mat,
                        const tensor_d& old_trans_mats, matrix_d& gammas);
        void fit_matrix(const vector_int &states, int h);
        void fit_matrix_states(const matrix_d &forw, const matrix_d &back, 
                        const vector_int &states, const matrix_d& old_hidden_mat,
                        const tensor_d& old_trans_mats);
        
        // Function needed to obtain probabilities needed for EM parameter estimation                      
        void forward(const vector_d &signal, const matrix_d &distr,
                        const matrix_d& init, const tensor_d& mats, 
                        const matrix_d& hidden_mat, tensor_d& forw);
        void forward_states(const vector_d &signal, const vector_int &states,
                        const matrix_d& init, const tensor_d& mats, 
                        const matrix_d& hidden_mat, matrix_d& forw);
        void backward(const vector_d &signal, const matrix_d &distr,
                        const matrix_d& init, const tensor_d& mats, 
                        const matrix_d& hidden_mat, tensor_d& back);
        void backward_states(const vector_d &signal, const vector_int &states,
                        const matrix_d& init, const tensor_d& mats, 
                        const matrix_d& hidden_mat, matrix_d& back);
        
};



#endif