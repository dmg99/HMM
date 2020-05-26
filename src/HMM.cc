#include <iostream>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <stdlib.h>
#include "HMM.hh"

using namespace std;
using vector_d = vector<double>;
using matrix_d = vector<vector_d>;
using tensor_d = vector<matrix_d>;
using vector_int = vector<int>;
using matrix_int = vector<vector_int>;
using tensor_int = vector<matrix_int>;

HMM::HMM(int s){
    set_nstates(s);
    vector_d uniforme(s, 1.0/s);
    set_initial_dist(uniforme);
    matrix_d def_params(2, vector_d(s, 0));
    def_params[1] = vector_d(s, 1);
    set_params(def_params);
}

HMM::HMM(int s, const matrix_d& mat, const matrix_d& params) {
    set_nstates (s);
    set_trans_mat (mat);
    set_params (params);
    vector_d uniforme(s, 1.0 / s);
    set_initial_dist(uniforme);
}

int HMM::get_nstates(){
    return nstates;
}

matrix_d HMM::get_params(){
    return parameters;
}

matrix_d HMM::get_trans_mat(){
    return trans_matrix;
}

vector_d HMM::get_initial_dist(){
    return initial_dist;
}

int HMM::get_decode_iter_print(){
    return decode_iter_print;
}

bool HMM::get_print_decode(){
    return print_decode;
}

bool HMM::get_pdf_constant(){
    return pdf_constant;
}

bool HMM::get_log_em(){
    return log_em;
}


void HMM::set_nstates(int s){
    nstates = s;
}

void HMM::set_params(const matrix_d& mat){
    parameters = mat;
}

void HMM::set_trans_mat(const matrix_d& mat){
    trans_matrix = mat;
}

void HMM::set_initial_matrix(){
    int N = get_nstates();
    matrix_d mat(N, vector_d(N,1));
    for (int n = 0; n < N; n++)
    {
        mat[n][n] += 1;
    }
    set_trans_mat(mat/(1.0*(N+1)));
}

void HMM::set_initial_random_matrix(){
    int N = get_nstates();
    matrix_d mat = random_mat(N, N);
    vector_d col_sums = sum(mat, 1);
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            mat[i][j] /= col_sums[j];
        }
    }
    set_trans_mat(mat);
}

void HMM::set_initial_dist(const vector_d& v){
    initial_dist = v;
}

void HMM::set_decode_iter_print(int iter){
    decode_iter_print = iter;
    print_decode = true;
}

void HMM::set_print_decode(bool p){
    print_decode = p;
}

void HMM::switch_log_em(){
    log_em = not log_em;
}

void HMM::switch_pdf_constant(){
    pdf_constant = not pdf_constant;
}

void HMM::set_initial_means(const vector_d& means){
    assert(means.size() == get_nstates());
    matrix_d pam = get_params();
    for (int i = 0; i < get_nstates(); ++i){
        pam[0][i] = means[i];
    }
    init_params = false;
    set_params(pam);
}

void HMM::set_initial_vars(const vector_d& vars){
    assert(vars.size() == get_nstates());
    matrix_d pam = get_params();
    for (int i = 0; i < get_nstates(); ++i){
        pam[1][i] = vars[i];
    }
    init_params = false;
    set_params(pam);
}

double HMM::gaussian_pdf(double x, double mu, double var){
    double factor = 1 / sqrt(2 * M_PI * var);
    double exponent = exp(-(x - mu)*(x - mu) / (2 * var));
    return factor * exponent;
}

// Matrix T x N: distribution of T observations on the N classes 
matrix_d HMM::distribution(const vector_d &signal){
    matrix_d params = get_params();
    int T = signal.size(),
        N = get_nstates();

    matrix_d distr(T, vector_d(N, 0));

    for (int t = 0; t < T; t++)
    {
        for (int n = 0; n < N; n++)
        {
            distr[t][n] = gaussian_pdf(signal[t], params[0][n], params[1][n]);
        }
    }

    return distr;
}

vector_d HMM::distribution(double observation){
    matrix_d params = get_params();

    int n = get_nstates();

    vector_d distr(n, 0);

    for (int j = 0; j < n; j++){
        distr[j] = gaussian_pdf(observation, params[0][j], params[1][j]);
    }

    return distr;
}



void HMM::gaussian_params(const vector_d& signal, const vector_int& v_states){
    int N = get_nstates(),
        T = signal.size();
    assert(signal.size()==v_states.size());

    vector_d count(N, 0);
    matrix_d sums(2, vector_d(N,0));
    for (int t = 0; t < T; t++)
    {
        int state = v_states[t];
        sums[0][state] += signal[t];
        sums[1][state] += signal[t]*signal[t];
        count[state] += 1;
    }

    print(count);

    matrix_d params(2, vector_d(N,0));
    for (int i = 0; i < N; i++)
    {
        if (count[i] > 0){
            params[0][i] = sums[0][i] / count[i];
            if (count[i] > 1) {
                double den = count[i] - 1;
                params[1][i] = sums[1][i] / den - count[i] * params[0][i] * params[0][i] / (den);
            }
        }
    }

    set_params(params);
}

void HMM::gaussian_params(const vector_d &signal, const matrix_d &gammes){
    int N = get_nstates(),
        T = signal.size();

    assert(signal.size() == gammes.size());

    vector_d tot_prob(N, 0.0001);
    matrix_d params(2, vector_d(N, 0));
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; ++n){
            params[0][n] += signal[t] * gammes[t][n];
            params[1][n] += signal[t] * signal[t] * gammes[t][n];
            tot_prob[n] += gammes[t][n];
        }
    }

    for (int n = 0; n < N; ++n){
        params[0][n] = params[0][n] / tot_prob[n];
        params[1][n] = params[1][n] / tot_prob[n];
        params[1][n] -= params[0][n]*params[0][n];
    }

    set_params(params);
}

void HMM::fit_matrix(const tensor_d &epsilons, const matrix_d &gammas){
    int T = gammas.size(),
        N = get_nstates();

    vector_d sums(N, 0);
    matrix_d trans(N, vector_d(N, 0));
    
    for (int t = 0; t < T-1; t++)
    {
        sums = sums + gammas[t];
        trans = trans + epsilons[t];
    }
    
    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < N; m++)
        {
            trans[n][m] /= sums[m];
        }
    }

    set_trans_mat(trans);
}

void HMM::fit_matrix_fast(const matrix_d &forw, const matrix_d &back, 
                    const matrix_d &distrib, const matrix_d &gammas)
{
    int T = gammas.size(),
        N = get_nstates();

    vector_d sums(N, 0);
    matrix_d trans(N, vector_d(N, 0));
    matrix_d epsilons(N, vector_d(N,0));

    for (int t = 0; t < T - 1; t++)
    {   
        epsilation(forw[t], back[t+1], distrib[t+1], epsilons);
        // sums += gammas[t];
        // trans += epsilons;
        for (int i = 0; i < N; i++)
        {
            sums[i] += gammas[t][i];
            for (int j = 0; j < N; j++)
            {
                trans[j][i] += epsilons[j][i];
            }
        }
    }

    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < N; m++)
        {
            trans[n][m] /= sums[m];
        }
    }

    set_trans_mat(trans);
}

void HMM::fit_matrix(const vector_int &states){
    int T = states.size(),
        N = get_nstates();

    matrix_d probs(T, vector_d(N, 0));

    for (int i = 0; i < T; ++i){
        int s = states[i];
        probs[i][s] = 1;
    }

    matrix_d temp_mat = probs;
    temp_mat.pop_back();
    matrix_d null(1, get_initial_dist());
    temp_mat = concat(null, temp_mat, 1);
    matrix_d prod = transpose(temp_mat) & probs;
    vector_d sums = sum(prod, 1);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (sums[j] > 1e-6)
                prod[i][j] /= sums[j];
            else
                prod[i][j] = 0;
        }
    }

    set_trans_mat(prod);
}

// Canviarho com la dabans!
void HMM::fit_matrix(const matrix_d &probs){
    int n = get_nstates();

    matrix_d temp_mat = probs;
    temp_mat.pop_back();
    temp_mat = concat(matrix_d(1, get_initial_dist()), temp_mat, 0);
    matrix_d prod = transpose(temp_mat) & probs;
    
    for (int i = 0; i < n; ++i)
    {
        double eps = 0.0001;
        double suma = sum(temp_mat[i]) + eps;
        for (int j = 0; j < n; ++j)
        {
            prod[i][j] /= suma;
        }
    }

    set_trans_mat(prod);
}

void HMM::fit_model_params(const vector_d& signal) {
    int T = signal.size(),
        N = get_nstates();

    matrix_d distr = distribution(signal);
    matrix_d forw = forward(signal, distr);
    matrix_d back = backward(signal, distr);
    matrix_d gammas = gammation(forw, back);
    tensor_d epsilons(T-1, matrix_d(N, vector_d(N, 0)));
    epsilation(forw, back, distr, epsilons);
    if (not get_pdf_constant()){
        gaussian_params(signal, gammas);
    }
    fit_matrix(epsilons, gammas);
}

void HMM::fit_model_params_fast(const vector_d &signal)
{
    matrix_d distr = distribution(signal);
    matrix_d forw = forward(signal, distr);
    matrix_d back = backward(signal, distr);
    matrix_d gammas;
    gammation_fast(forw, back, gammas);
    if (not get_pdf_constant())
    {
        gaussian_params(signal, gammas);
    }
    fit_matrix_fast(forw, back, distr, gammas);
}

void HMM::fit_model_params_from_truth(const vector_d &signal, const vector_int &v_states){

    cout << "Ditsribution no" << endl;
    cout << get_pdf_constant() << endl;
    if (not get_pdf_constant())
    {
        gaussian_params(signal, v_states);
    }
    fit_matrix(v_states);
}

void HMM::fit_model_params(const vector_int &v_states){
    fit_matrix(v_states);
}

void HMM::fit_model_params(const matrix_d &probs){
    fit_matrix(probs);
}



// Returns probability from state i in instant t-1 to state j in instant t
matrix_d HMM::path_probs(const matrix_d &matrix, const vector_d &curr_prob,
                         const vector_d &distr)
{
    matrix_d result = dyadic_product(curr_prob, distr);
    result = matrix * result;

    return result;
}

// Given a set of observations, computes the probability that have been generated by this model
double HMM::likelihood(const vector_d& signal){
    int T = signal.size(),
        N = get_nstates();
    vector_d initial = get_initial_dist();
    matrix_d mat = get_trans_mat();

    if (T==0) return 0;

    // Contains likelihood of class j up to iteration i
    matrix_d probs(T, vector_d(N,0));
    probs[0] = initial * distribution(signal[0]);

    for (int t = 1; t < T; t++){
        probs[t] = (mat & probs[t-1]) * distribution(signal[t]);
    }

    return sum(probs[T-1]);
}


vector_int HMM::decode_seq(const int states, const matrix_d &matrix,
                      const vector_d &initial_distr, const matrix_d &params,
                      const matrix_d &distr, const bool prints,
                      const int iter_print)
{
    int T = distr.size();
    
    //matrix_int paths(states, vector_int(T));
    //matrix_int new_path(states, vector_int(T));

    vector_d curr_prob(states, 0);
    matrix_int prev_state(T, vector_int(states, 0));
    double suma = 0;
    for (int i = 0; i < states; ++i)
    {
        curr_prob[i] = initial_distr[i] * distr[0][i];
        suma += curr_prob[i];
    }
    curr_prob = curr_prob  / suma;
    /*
    for (int i = 0; i < states; ++i)
        paths[0][i] = i;
    */
    // for to iterate over all time instants
    for (int t = 1; t < T; ++t)
    {
        // Aij * prob(i) * prob(output)
        matrix_d new_probs = path_probs(matrix, curr_prob, distr[t]);
        if (prints and t % iter_print == 0)
            cout << "Viterbi iterations: " << t << endl;

        pair<vector_int, vector_d> maxs = argmax(new_probs, 1);
        //vector_int max_temp = maxs.first;
        prev_state[t] = maxs.first;
        curr_prob = maxs.second;
        curr_prob = curr_prob * (1 / sum(curr_prob));
        /*
        for (int s = 0; s < states; ++s)
        {
            int prev = max_temp[s];
            new_path[s] = paths[prev];
            new_path[s][t] = s;
        }
        // No cunde
        paths = new_path;
        */
    }

    cout << "Total Viterbi iterations: " << T << endl;

    int final_max = argmax(curr_prob).first;
    vector_int decoded_seq(T, 0);
    decoded_seq[T-1] = final_max;
    for (int t = T-2; t >= 0; --t){
        int previous = decoded_seq[t+1];
        decoded_seq[t] = prev_state[t+1][previous];
    }

    //return paths[final_max];
    return decoded_seq;
}

vector_int HMM::decode_prob(const int states, const matrix_d &matrix,
                           const vector_d &initial_distr, const matrix_d &params,
                           const matrix_d &distr, const bool prints,
                           const int iter_print)
{
    int T = distr.size();

    matrix_int paths(states, vector_int(T));
    matrix_int new_path(states, vector_int(T));

    matrix_d probs(T, vector_d(states));

    probs[0] = initial_distr * distr[0];
    probs[0] = probs[0] * (1 / sum(probs[0]));

    for (int i = 0; i < states; ++i)
        paths[0][i] = i;

    // for to iterate over all time instants
    for (int t = 1; t < T; ++t)
    {

        // Aij * prob(i) * prob(output)
        matrix_d new_probs = path_probs(matrix, probs[t], distr[t]);

        if (prints and t % iter_print == 0)
            cout << "Viterbi iterations: " << t << endl;

        pair<vector_int, vector_d> maxs = argmax(new_probs, 0);
        vector_int max_temp = maxs.first;
        probs[t] = maxs.second;
        probs[t] = probs[t] * (1 / sum(probs[t]));

        for (int s = 0; s < states; ++s)
        {
            int prev = max_temp[s];
            new_path[s] = paths[prev];
            new_path[s][t] = s;
        }

        paths = new_path;
    }

    cout << "Total Viterbi iterations: " << T << endl;

    int final_max = argmax(probs[T-1]).first;

    return paths[final_max];
}

vector_int HMM::decode(const vector_d &signal, bool only_seq){
    matrix_d distr = distribution(signal);
    assert(only_seq);
    return decode_seq(get_nstates(), get_trans_mat(), get_initial_dist(),
                      get_params(), distr, get_print_decode(),
                      get_decode_iter_print());
}


matrix_d HMM::forward(const vector_d &signal, const matrix_d &distr){
    cout << "Forward" << endl;
    vector_d init = get_initial_dist();
    matrix_d mat = get_trans_mat();
    int T = signal.size(),
        N = get_nstates();

    matrix_d forw(T, vector_d(N, 0));
    forw[0] = init * distr[0];
    forw[0] = forw[0]/sum(forw[0]);
    for (int t = 1; t < T; t++){
        forw[t] = distr[t] * (mat & forw[t-1]);
        forw[t] = forw[t] * (1/sum(forw[t]));
    }
    return forw;
}

// To be tested
matrix_d HMM::forward_fast(const vector_d &signal, const matrix_d &distr){
    cout << "Forward" << endl;
    vector_d init = get_initial_dist();
    matrix_d mat = get_trans_mat();
    int T = signal.size(),
        N = get_nstates();

    matrix_d forw = distr;
    double sum_init = 0;
    for (int i = 0; i < N; ++i){
        forw[0][i] *= init[i];
        sum_init += forw[0][i];
    }
    for (int i = 0; i < N; ++i){
        forw[0][i] /= sum_init;
    }

    for (int t = 1; t < T; t++){
        vector_d mult_vec = mat & forw[t-1];
        double curr_sum = 0;
        for (int i = 0; i < N; ++i){
            forw[t][i] *= mult_vec[i];
            curr_sum += forw[t][i];
        }
        for (int i = 0; i < N; ++i){
            forw[t][i] /= curr_sum;
        }
    }
    return forw;
}

matrix_d HMM::backward(const vector_d &signal, const matrix_d &distr){
    cout << "Backward" << endl;
    vector_d init = get_initial_dist();
    matrix_d mat = get_trans_mat();
    int T = signal.size(),
        N = get_nstates();

    matrix_d back(T, vector_d(N, 1.0/N));
    for (int t = T-2; t >=0; t--){
        back[t] = (distr[t+1] * back[t+1]) & mat;
        back[t] = back[t] * (1/sum(back[t]));
    }
    return back;
}

// Pasar sortida per referencia si cal (falta testejar)
matrix_d HMM::backward_fast(const vector_d &signal, const matrix_d &distr){
    cout << "Backward" << endl;
    vector_d init = get_initial_dist();
    matrix_d mat = get_trans_mat();
    int T = signal.size(),
        N = get_nstates();

    matrix_d back(T, vector_d(N, 1.0/N));
    for (int t = T-2; t >=0; t--){
        back[t] = (distr[t+1] * back[t+1]) & mat;
        double curr_sum = sum(back[t]);
        for (int n = 0; n < N; ++n)
            back[t][n] /= curr_sum;
    }
    return back;
}

matrix_d HMM::gammation(const matrix_d &forw, const matrix_d &back)
{
    cout << "Gammation" << endl;
    int T = forw.size(),
        N = get_nstates();

    matrix_d gamma(T, vector_d(N, 0));
    for (int t = 0; t < T; ++t){
        gamma[t] = forw[t] * back[t];
        gamma[t] = gamma[t]/sum(gamma[t]);
    }
    return gamma;
}

void HMM::gammation_fast(const matrix_d &forw, const matrix_d &back,
                             matrix_d &gammas)
{
    cout << "Gammation" << endl;
    int T = forw.size(),
        N = get_nstates();

    gammas = forw;
    for (int t = 0; t < T; ++t)
    {
        double suma = 0;
        // gamma[t] = forw[t] * back[t];
        for (int i = 0; i < N; i++)
        {
            gammas[t][i] *= back[t][i];
            suma += gammas[t][i];
        }
        for (int i = 0; i < N; i++)
        {
            gammas[t][i] /= suma;
        }
        
    }
    return;
}

void HMM::epsilation(const matrix_d &forw, const matrix_d &back, 
                         const matrix_d &distr, tensor_d &epsilons)
{
    cout << "Epsilation" << endl;
    int T = forw.size();

    matrix_d mat = get_trans_mat();

    for (int t = 0; t < T-1; t++)
    {
        epsilons[t] = mat * dyadic_product(distr[t + 1] * back[t + 1], forw[t]);
        epsilons[t] = epsilons[t] / sum(epsilons[t]);
    }

    cout << "epsilation ha acabat" << endl;
    return;
}

void HMM::epsilation(const vector_d &forw, const vector_d &back, 
                     const vector_d &distrib, matrix_d &epsilons){
    epsilons = get_trans_mat();
    int N = get_nstates();
    double sum = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {//epsilation(forw[t], back[t+1], distrib[t+1], epsilons);
            epsilons[j][i] *= forw[i] * distrib[j] * back[j];
            sum += epsilons[j][i];
        }
    }

    // Hauria de ser aixo:
    // epsilons /= sum(epsilons);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            epsilons[j][i] /= sum;
        }
        
    }
}

void HMM::EM_maximization(const vector_d &signal, const vector_int &states,
                          double tol, int maxits, int seed, int iter_print)
{

    int N = get_nstates();

    if (init_params){
        if (get_pdf_constant())
        {
            assert(signal.size() == states.size());
            gaussian_params(signal, states);
        }

        else
        {
            matrix_d pams(2, vector_d(N, 1));
            double max = argmax(signal).second,
            min = argmin(signal).second;
            for (int n = 0; n < N; n++)
                pams[0][n] = ((max - min) * n) / (2.0 * (N - 1)) + min + (max - min)/4;

            set_params(pams);
        }
    }

    if (seed <= 0)
        set_initial_matrix();
    else{
        srand(seed);
        set_initial_random_matrix();
    }
    

    double eps = tol + 1;
    for (int it=0; abs(eps) > tol and it < maxits; ++it){

        matrix_d old_params = get_params();
        matrix_d old_mat = get_trans_mat();
        fit_model_params_fast(signal);
        if (get_log_em()) {
            cout << "Iteration: " << it << endl;
            cout << "PARAMETERS" << endl;
            print(get_params());
            cout << "MATRIU DE TRANSICIÓ" << endl;
            print(get_trans_mat());
        }
        
        eps = norm(get_params() - old_params) + norm(get_trans_mat() - old_mat);
        if (it % iter_print == 0){
            cout << eps << endl;
        }
    }

    cout << "PARAMETERS FINALS" << endl;
    print(get_params());
    cout << "MATRIU DE TRANSICIÓ" << endl;
    print(get_trans_mat());
}

void HMM::EM_maximization_vit(const vector_d &signal, double tol, 
                              int maxits, int iter_print){
    double eps = tol + 1;

    for (int it = 0; abs(eps) and it < maxits; ++it)
    {
        if (it % iter_print == 0)
            cout << "Iteration: " << it << endl;

        matrix_d old_params = get_params();
        vector_int states = decode(signal, true);
        fit_model_params_from_truth(signal, states);
        eps = norm(get_params() - old_params);
        cout << eps << endl;
    }
}