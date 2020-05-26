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
#include "HiddenHMM.hh"

using namespace std;
using vector_d = vector<double>;
using matrix_d = vector<vector_d>;
using tensor_d = vector<matrix_d>;
using vector_int = vector<int>;
using matrix_int = vector<vector_int>;
using tensor_int = vector<matrix_int>;

HiddenHMM::HiddenHMM(int s, int h){
    set_nstates(s);
    set_nhidden(h);

    matrix_d uniforme(h, vector_d(s, 1.0/(s*h)));
    set_initial_dist(uniforme);

    matrix_d def_params(2, vector_d(s, 0));
    def_params[1] = vector_d(s, 1);
    set_params(def_params);

    tensor_d init_mats(h, matrix_d(s, vector_d(s)));
    set_trans_mat(init_mats);
}

int HiddenHMM::get_nstates(){
    return nstates;
}

int HiddenHMM::get_nhidden(){
    return nhidden;
}

matrix_d HiddenHMM::get_params(){
    return parameters;
}

matrix_d HiddenHMM::get_trans_mat(int h){
    assert(h < get_nhidden());
    return trans_matrix[h];
}

tensor_d HiddenHMM::get_trans_mat(){
    return trans_matrix;
}

matrix_d HiddenHMM::get_hidden_mat(){
    return hidden_matrix;
}

vector_d HiddenHMM::get_initial_dist(int h){
    assert(h < get_nhidden());
    return initial_dist[h];
}

matrix_d HiddenHMM::get_initial_dist(){
    return initial_dist;
}

int HiddenHMM::get_decode_iter_print(){
    return decode_iter_print;
}

bool HiddenHMM::get_print_decode(){
    return print_decode;
}

bool HiddenHMM::get_pdf_constant(){
    return pdf_constant;
}

bool HiddenHMM::get_log_em(){
    return log_em;
}


void HiddenHMM::set_nstates(int s){
    nstates = s;
}

void HiddenHMM::set_nhidden(int h){
    nhidden = h;
}

void HiddenHMM::set_params(const matrix_d& mat){
    parameters = mat;
}


void HiddenHMM::set_trans_mat(const tensor_d& mats){
    assert(mats.size() == get_nhidden());
    assert(mats[0].size() == get_nstates());
    assert(mats[0][0].size() == get_nstates());
    trans_matrix = mats;
}

void HiddenHMM::set_trans_mat(const matrix_d& mat, int h){
    assert(h < get_nhidden());
    trans_matrix[h] = mat;
}

void HiddenHMM::set_hidden_mat(const matrix_d& mat){
    hidden_matrix = mat;
}


void HiddenHMM::set_initial_matrix(int h, int add){
    assert(h < get_nhidden());
    assert(add >= -1);
    int N = get_nstates();
    matrix_d mat(N, vector_d(N,1));
    for (int n = 0; n < N; n++)
    {
        mat[n][n] += add;
    }
    set_trans_mat(mat/(1.0*N + add), h);
}

void HiddenHMM::set_initial_random_matrix(int h){
    int N = get_nstates(),
        H = get_nhidden();

    assert(h < H);
    matrix_d mat = random_mat(N, N);
    vector_d col_sums = sum(mat, 1);
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            mat[i][j] /= col_sums[j];
        }
    }
    set_trans_mat(mat, h);
}

void HiddenHMM::set_initial_hidden(){
    int H = get_nhidden();
    matrix_d mat(H, vector_d(H,1));
    for (int n = 0; n < H; n++)
    {
        mat[n][n] += 1;
    }
    set_hidden_mat(mat/(1.0*H + 1.0));
}

void HiddenHMM::set_initial_dist(const vector_d& v, int h){
    assert(h < get_nhidden());
    initial_dist[h] = v;
}

void HiddenHMM::set_initial_dist(const matrix_d& m){
    assert(m.size() == get_nhidden());
    assert(m[0].size() == get_nstates());
    initial_dist = m;
}

void HiddenHMM::set_initial_dist(const vector_d& hidden_dist, const vector_d& state_dist){
    int H = get_nhidden(),
        N = get_nstates();
    assert(hidden_dist.size() == H);
    assert(state_dist.size() == N);
    matrix_d initial_dist(H, vector_d(N, 0));
    for (int i = 0; i < H; ++i){
        for (int j = 0; j < N; ++j){
            initial_dist[i][j] = hidden_dist[i]*state_dist[j];
        }
    }
    set_initial_dist(initial_dist);
}

void HiddenHMM::set_decode_iter_print(int iter){
    decode_iter_print = iter;
    print_decode = true;
}

void HiddenHMM::set_print_decode(bool p){
    print_decode = p;
}


void HiddenHMM::switch_log_em(){
    log_em = not log_em;
}

void HiddenHMM::switch_init(){
    init_params = not init_params;
}

void HiddenHMM::switch_pdf_constant(){
    pdf_constant = not pdf_constant;
}

void HiddenHMM::switch_state_training(){
    state_training = not state_training;
}

void HiddenHMM::set_initial_means(const vector_d& means){
    assert(means.size() == get_nstates());
    matrix_d pam = get_params();
    for (int i = 0; i < get_nstates(); ++i){
        pam[0][i] = means[i];
    }
    init_params = false;
    set_params(pam);
}

void HiddenHMM::set_initial_vars(const vector_d& vars){
    assert(vars.size() == get_nstates());
    matrix_d pam = get_params();
    for (int i = 0; i < get_nstates(); ++i){
        pam[1][i] = vars[i];
    }
    init_params = false;
    set_params(pam);
}


double HiddenHMM::gaussian_pdf(double x, double mu, double var){
    double factor = 1 / sqrt(2 * M_PI * var);
    double exponent = exp(-(x - mu)*(x - mu) / (2 * var));
    return factor * exponent;
}

// Matrix T x N: distribution of T observations on the N classes 

matrix_d HiddenHMM::distribution(const vector_d &signal){
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

vector_d HiddenHMM::distribution(double observation){
    matrix_d params = get_params();
    int n = get_nstates();

    vector_d distr(n, 0);
    for (int j = 0; j < n; j++){
        distr[j] = gaussian_pdf(observation, params[0][j], params[1][j]);
    }

    return distr;
}


void HiddenHMM::gaussian_params(const vector_d& signal, const vector_int& v_states){
    int N = get_nstates(),
        T = signal.size();

    assert(signal.size() == v_states.size());

    vector_d count(N, 0);
    matrix_d sums(2, vector_d(N,0));
    for (int t = 0; t < T; t++)
    {
        int state = v_states[t];
        sums[0][state] += signal[t];
        sums[1][state] += signal[t]*signal[t];
        count[state] += 1;
    }

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

void HiddenHMM::gaussian_params(const vector_d &signal, const matrix_d &gammes){
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

void HiddenHMM::fit_matrix(const tensor_d &forw, const tensor_d &back, 
                        const matrix_d &distrib, const matrix_d& old_hidden_mat,
                        const tensor_d& old_trans_mats, matrix_d & gammas)
{
    int T = forw.size(),
        N = get_nstates(),
        H = get_nhidden();

    matrix_d trans_sums(H, vector_d(N, 0));
    vector_d hidden_sums(H, 0);
    matrix_d hidden_mat(H, vector_d(H, 0));
    tensor_d trans_mats(H, matrix_d(N, vector_d(N, 0)));

    for (int t = 0; t < T-1; t++)
    {   
        tensor_d aux_trans_mats(H, matrix_d(N, vector_d(N, 0)));
        matrix_d aux_hidden_mat(H, vector_d(H, 0));
        
        double total_it_sum = 0;
        for (int h = 0; h < H; ++h){
            for(int g = 0; g < H; ++g){
                // Hidden_update is epsilon(t)
                double paths_sum = 0;
                for(int i = 0; i < N; ++i){
                    for(int j = 0; j < N; ++j){
                        // Revisar
                        double curr_epsilon = forw[t][g][i]*old_hidden_mat[h][g]*old_trans_mats[h][j][i]*
                                              distrib[t+1][j]*back[t+1][h][j];
                        aux_trans_mats[h][j][i] += curr_epsilon;  
                        paths_sum += curr_epsilon;
                        total_it_sum += curr_epsilon;
                        gammas[t][i] += curr_epsilon;
                    }
                }
                aux_hidden_mat[h][g] = paths_sum;
            }
        }
        // Standarize
        for (int h = 0; h < H; ++h){
            for(int i = 0; i < N; ++i){
                for(int j = 0; j < N; ++j){
                    double update = aux_trans_mats[h][j][i]/total_it_sum;
                    trans_mats[h][j][i] += update;
                    trans_sums[h][i] += update;
                }
            }
            for (int g = 0; g < H; ++g){
                double update = aux_hidden_mat[h][g]/total_it_sum;
                hidden_mat[h][g] += update;
                hidden_sums[g] += update;
            }
        }
        for (int i = 0; i < N; ++i)
            gammas[t][i] /= total_it_sum;
    }
    
    // Standarize everything
    for (int h = 0; h < H; ++h){
        for (int g = 0; g < H; ++g)
            hidden_mat[h][g] /= hidden_sums[g];
        
        for (int i = 0; i < N; ++i){
            for (int j = 0; j < N; ++j)
                trans_mats[h][j][i] /= trans_sums[h][i];
        }
    }

    set_trans_mat(trans_mats);
    set_hidden_mat(hidden_mat);
}

void HiddenHMM::fit_matrix_states(const matrix_d &forw, const matrix_d &back, 
                        const vector_int &states, const matrix_d& old_hidden_mat,
                        const tensor_d& old_trans_mats)
{
    cout << "Fit Matrix States" << endl;
    int T = forw.size(),
        N = get_nstates(),
        H = get_nhidden();

    matrix_d trans_sums(H, vector_d(N, 0));
    vector_d hidden_sums(H, 0);
    matrix_d hidden_mat(H, vector_d(H, 0));
    tensor_d trans_mats(H, matrix_d(N, vector_d(N, 0)));    

    for (int t = 0; t < T-1; t++)
    {   
        vector_d aux_trans_mats(H,  0);
        matrix_d aux_hidden_mat(H, vector_d(H, 0));
        
        int i = states[t],
            j = states[t+1];

        double total_it_sum = 0;
        for (int h = 0; h < H; ++h){
            for(int g = 0; g < H; ++g){
                // Hidden_update is epsilon(t)
                double curr_epsilon = forw[t][g]*old_hidden_mat[h][g]*
                                      old_trans_mats[h][j][i]*back[t+1][h];
                aux_trans_mats[h] += curr_epsilon;  
                aux_hidden_mat[h][g] += curr_epsilon;
                total_it_sum += curr_epsilon;
            }
        }
        // Standarize
        for (int h = 0; h < H; ++h){
            double update = aux_trans_mats[h]/total_it_sum;
            trans_mats[h][j][i] += update;
            trans_sums[h][i] += update;
            for (int g = 0; g < H; ++g){
                double update = aux_hidden_mat[h][g]/total_it_sum;
                hidden_mat[h][g] += update;
                hidden_sums[g] += update;
            }
        }
    }

    double eps = 1e-8;
    // Standarize everything
    for (int h = 0; h < H; ++h){
        for (int i = 0; i < N; ++i){
            for (int j = 0; j < N; ++j)
                trans_mats[h][j][i] /= trans_sums[h][i] + eps;
        }
        for (int g = 0; g < H; ++g)
                hidden_mat[h][g] /= hidden_sums[g] + eps;

    }

    set_trans_mat(trans_mats);
    set_hidden_mat(hidden_mat);
    
    return;
}

// Upsi
void HiddenHMM::fit_matrix(const vector_int &states, int h){
    int T = states.size(),
        N = get_nstates();

    matrix_d probs(T, vector_d(N, 0));

    for (int i = 0; i < T; ++i){
        int s = states[i];
        probs[i][s] = 1;
    }

    matrix_d temp_mat = probs;
    temp_mat.pop_back();
    matrix_d null(1, get_initial_dist(h));
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

    set_trans_mat(prod, h);
}


void HiddenHMM::fit_model_params(const vector_d& signal, const matrix_d& hidden_mat,
                        const tensor_d& trans_mats) {
    int T = signal.size(),
        N = get_nstates(),
        H = get_nhidden();

    // Get parameters that will be used many times
    matrix_d distr = distribution(signal);
    matrix_d init_dist = get_initial_dist();
    
    // Get forward probs
    tensor_d forw(T, matrix_d(H, vector_d(N, 0)));
    forward(signal, distr, init_dist, trans_mats, hidden_mat, forw);
    // Get backward probs
    tensor_d back(T, matrix_d(H, vector_d(N, 1)));
    backward(signal, distr, init_dist, trans_mats, hidden_mat, back);

    matrix_d gammas(T, vector_d(N, 0));
    fit_matrix(forw, back, distr, hidden_mat, trans_mats, gammas);
    if (not get_pdf_constant()){
        gaussian_params(signal, gammas);
    }
}

void HiddenHMM::fit_model_params_states(const vector_d& signal, const vector_int &states, 
                                    const matrix_d& hidden_mat, const tensor_d& trans_mats) {
    int T = signal.size(),
        N = get_nstates(),
        H = get_nhidden();

    // Get parameters that will be used many times
    matrix_d init_dist = get_initial_dist();
    
    // Get forward probs
    matrix_d forw(T, vector_d(H, 0));
    forward_states(signal, states, init_dist, trans_mats, hidden_mat, forw);
    // Get backward probs
    matrix_d back(T, vector_d(H, 1));
    backward_states(signal, states, init_dist, trans_mats, hidden_mat, back);

    fit_matrix_states(forw, back, states, hidden_mat, trans_mats);
}

void HiddenHMM::fit_model_params_from_truth(const vector_d &signal, const vector_int &v_states){
    gaussian_params(signal, v_states);
}


matrix_int HiddenHMM::decode_seq(const int& S, const int& H, const matrix_d& hidden_mat,
                    const tensor_d &matrices, const matrix_d &initial_distr, const matrix_d &params,
                    const matrix_d &distr, const bool prints,
                    const int iter_print)
{
    int T = distr.size();
    matrix_d curr_prob(H, vector_d(S, 0));
    
    // Size T x H x S x 2 (it stores 2 cordinates)
    vector<vector<vector<pair<int, int>>>> prev_state(T, vector<vector<pair<int, int>>>(H, vector<pair<int, int>>(S)));
    // Compute initial probs
    double sum = 0;
    for(int h = 0; h < H; ++h){
        for (int i = 0; i < S; ++i){
            for(int g = 0; g < H; ++g){
                for(int j = 0; j < S; ++j){
                    curr_prob[h][i] += initial_distr[g][j]*hidden_mat[h][g]*matrices[h][i][j]* distr[0][i];
                }
            }
            sum += curr_prob[h][i];
        }
    }
    curr_prob = curr_prob / sum;
    // for to iterate over all time instants
    for (int t = 1; t < T; ++t)
    {
        if (prints and t % iter_print == 0)
            cout << "Viterbi iterations: " << t << endl;

        matrix_d new_probs(H, vector_d(S, 0));
        double curr_sum = 0;
        for (int h = 0; h < H; ++h){
            for(int i = 0; i < S; ++i){
                // Calculate highest prob path to (hidden h, state j)
                double highest_prob = 0;
                pair<int, int> best_path;
                for(int g = 0; g < H; ++g){
                    for(int j = 0; j < S; ++j){
                        double path_prob = hidden_mat[h][g]*matrices[h][i][j]*curr_prob[g][j]*distr[t][i];
                        if (path_prob > highest_prob){
                            highest_prob = path_prob;
                            best_path.first = g;
                            best_path.second = j;
                        }
                    }
                }
                curr_sum += highest_prob;
                new_probs[h][i] = highest_prob;
                prev_state[t][h][i] = best_path;
            }
        }

        // Assign to curr_probs and divide by sum
        for (int h = 0; h < H; ++h){
            for(int i = 0; i < S; ++i)
                curr_prob[h][i] = new_probs[h][i]/curr_sum;
        }
    }

    cout << "Total Viterbi iterations: " << T << endl;

    pair<int, int> final_max = mat_argmax(curr_prob);
    vector_int decoded_seq(T, 0);
    vector_int decoded_hidden(T, 0);
    
    decoded_seq[T-1] = final_max.second;
    decoded_hidden[T-1] = final_max.first;
    
    for (int t = T-2; t >= 0; --t){
        int prev_s = decoded_seq[t+1];
        int prev_h = decoded_hidden[t+1];
        
        decoded_seq[t]    = prev_state[t+1][prev_h][prev_s].second;
        decoded_hidden[t] = prev_state[t+1][prev_h][prev_s].first;
    }
    
    // Es podria evitar copia, pero no ve d'aixo
    matrix_int decoded_all(2);
    decoded_all[0] = decoded_seq;
    decoded_all[1] = decoded_hidden;
    
    return decoded_all;
}

tensor_d HiddenHMM::decode_seq_probs(const int& S, const int& H, const matrix_d& hidden_mat,
                    const tensor_d &matrices, const matrix_d &initial_distr, const matrix_d &params,
                    const matrix_d &distr, const bool prints,
                    const int iter_print)
{
    int T = distr.size();
    tensor_d all_probs(T, matrix_d(H, vector_d(S, 0)));
    
    // Size T x H x S x 2 (it stores 2 cordinates)
    vector<vector<vector<pair<int, int>>>> prev_state(T, vector<vector<pair<int, int>>>(H, vector<pair<int, int>>(S)));
    // Compute initial probs
    double sum = 0;
    for(int h = 0; h < H; ++h){
        for (int i = 0; i < S; ++i){
            for(int g = 0; g < H; ++g){
                for(int j = 0; j < S; ++j){
                    all_probs[0][h][i] += initial_distr[g][j]*hidden_mat[h][g]*matrices[h][i][j]* distr[0][i];
                }
            }
            sum += all_probs[0][h][i];
        }
    }
    all_probs[0] = all_probs[0] / sum;
    // for to iterate over all time instants
    for (int t = 1; t < T; ++t)
    {
        if (prints and t % iter_print == 0)
            cout << "Viterbi iterations: " << t << endl;

        double curr_sum = 0;
        for (int h = 0; h < H; ++h){
            for(int i = 0; i < S; ++i){
                // Calculate highest prob path to (hidden h, state j)
                double highest_prob = 0;
                pair<int, int> best_path;
                for(int g = 0; g < H; ++g){
                    for(int j = 0; j < S; ++j){
                        double path_prob = hidden_mat[h][g]*matrices[h][i][j]*all_probs[t-1][g][j]*distr[t][i];
                        if (path_prob > highest_prob){
                            highest_prob = path_prob;
                            best_path.first = g;
                            best_path.second = j;
                        }
                    }
                }
                curr_sum += highest_prob;
                all_probs[t][h][i] = highest_prob;
                prev_state[t][h][i] = best_path;
            }
        }

        // Assign to curr_probs and divide by sum
        for (int h = 0; h < H; ++h){
            for(int i = 0; i < S; ++i)
                all_probs[t][h][i] = all_probs[t][h][i]/curr_sum;
        }
    }

    cout << "Total Viterbi iterations: " << T << endl;
    //pair<tensor_d, vector<vector<vector<pair<int, int>>>>> res;
    //res.first = all_probs;
    //res.second = prev_state;
    return all_probs;
}

matrix_int HiddenHMM::decode(const vector_d &signal, bool only_seq){
    matrix_d distr = distribution(signal);
    assert(only_seq);
    
    return decode_seq(get_nstates(), get_nhidden(), get_hidden_mat(), 
                      get_trans_mat(), get_initial_dist(),
                      get_params(), distr, get_print_decode(),
                      get_decode_iter_print());
    
}

tensor_d HiddenHMM::decode_probs(const vector_d &signal){
    matrix_d distr = distribution(signal);
    
    return decode_seq_probs(get_nstates(), get_nhidden(), get_hidden_mat(), 
                      get_trans_mat(), get_initial_dist(),
                      get_params(), distr, get_print_decode(),
                      get_decode_iter_print());
}

void HiddenHMM::forward(const vector_d &signal, const matrix_d &distr,
                        const matrix_d& init, const tensor_d& mats, 
                        const matrix_d& hidden_mat, tensor_d& forw)
{
    cout << "Forward" << endl;

    int T = signal.size(),
        N = get_nstates(),
        H = get_nhidden();

    // Calculate initial values
    double init_sum = 0;
    for (int h = 0; h < H; ++h){
        for(int i = 0; i < N; ++i){
            double current_prob = 0; 
            for(int g = 0; g < H; ++g){
                for(int j = 0; j < N; ++j){
                    current_prob += init[g][j]*hidden_mat[h][g]*mats[h][i][j]*distr[0][i];
                }
            }
            forw[0][h][i] = current_prob;
            init_sum += current_prob;
        }  
    }

    // Divide by sum
    for (int h = 0; h < H; ++h){
        for(int i = 0; i < N; ++i){
            forw[0][h][i] /= init_sum;
        }  
    }

    for (int t = 1; t < T; t++){
        double curr_sum = 0;
        for (int h = 0; h < H; ++h){
            for (int i = 0; i < N; ++i){
                // Sum all possible paths
                double curr_prob = 0;
                for (int g = 0; g < H; ++g){
                    for (int j = 0; j < N; ++j){
                        curr_prob += hidden_mat[h][g]*mats[h][i][j]*forw[t-1][g][j];
                    }
                }
                curr_prob *= distr[t][i];
                forw[t][h][i] = curr_prob;
                curr_sum += curr_prob;
            }
        }
        // Divide by sum
        for (int h = 0; h < H; ++h){
            for (int i = 0; i < N; ++i){
                forw[t][h][i] /= curr_sum;
            }
        }
    }
}

void HiddenHMM::backward(const vector_d &signal, const matrix_d &distr,
                        const matrix_d& init, const tensor_d& mats, 
                        const matrix_d& hidden_mat, tensor_d& back){
    cout << "Backward" << endl;
    int T = signal.size(),
        N = get_nstates(),
        H = get_nhidden();

    for (int t = T-2; t >=0; t--){
        double curr_sum = 0;
        for (int h = 0; h < H; ++h){
            for (int i = 0; i < N; ++i){
                // Sum all possible paths
                double curr_prob = 0;
                for (int g = 0; g < H; ++g){
                    for (int j = 0; j < N; ++j){
                        curr_prob += hidden_mat[g][h]*mats[g][j][i]*back[t+1][g][j]*distr[t+1][j];
                    }
                }
                back[t][h][i] = curr_prob;
                curr_sum += curr_prob;
            }
        }
        // Divide by sum
        for (int h = 0; h < H; ++h){
            for (int i = 0; i < N; ++i){
                back[t][h][i] /= curr_sum;
            }
        }
    }
}

void HiddenHMM::forward_states(const vector_d &signal, const vector_int &states,
                        const matrix_d& init, const tensor_d& mats, 
                        const matrix_d& hidden_mat, matrix_d& forw)
{
    cout << "Forward States" << endl;

    int T = signal.size(),
        N = get_nstates(),
        H = get_nhidden();

    // Calculate initial values
    double init_sum = 0;
    for (int i = states[0], h = 0; h < H; ++h){
        double current_prob = 0; 
        for(int g = 0; g < H; ++g){
            for(int j = 0; j < N; ++j){
                current_prob += init[g][j]*hidden_mat[h][g]*mats[h][i][j];
            }
        }
        forw[0][h] = current_prob;
        init_sum += current_prob;
    }

    // Divide by sum
    for (int h = 0; h < H; ++h){
        forw[0][h] /= init_sum;
    }

    for (int t = 1; t < T; t++){
        int i = states[t],
            j = states[t-1];
        double curr_sum = 0;
        for (int h = 0; h < H; ++h){
            // Sum all possible paths
            double curr_prob = 0;
            for (int g = 0; g < H; ++g){
                curr_prob += forw[t-1][g]*hidden_mat[h][g]*mats[h][i][j];
            }
            forw[t][h] = curr_prob;
            curr_sum += curr_prob;
        }
        // Divide by sum
        for (int h = 0; h < H; ++h){
            forw[t][h] /= curr_sum;
        }
    }
}

void HiddenHMM::backward_states(const vector_d &signal, const vector_int &states,
                        const matrix_d& init, const tensor_d& mats, 
                        const matrix_d& hidden_mat, matrix_d& back){
    cout << "Backward States" << endl;
    int T = signal.size(),
        N = get_nstates(),
        H = get_nhidden();

    for (int h = 0; h < H; ++h){
        back[T-1][h] = 1;
    }

    for (int t = T-2; t >=0; t--){
        double curr_sum = 0;
        int i = states[t],
            j = states[t+1];

        for (int h = 0; h < H; ++h){
            // Sum all possible paths
            double curr_prob = 0;
            for (int g = 0; g < H; ++g){
                curr_prob += hidden_mat[g][h]*mats[g][j][i]*back[t+1][g];
            }
            back[t][h] = curr_prob;
            curr_sum += curr_prob;
        }
        // Divide by sum
        for (int h = 0; h < H; ++h){
            back[t][h] /= curr_sum;
        }
    }
}

/*
void HiddenHMM::EM_maximization(const vector_d &signal, const vector_int &states,
                                double tol, int maxits, int seed, int iter_print){

    int N = get_nstates(),
        H = get_nhidden();

    if (init_params){
        if (get_pdf_constant()){
            assert(signal.size() == states.size());
            gaussian_params(signal, states);
        }
        else{
            matrix_d pams(2, vector_d(N, 1));
            double max = argmax(signal).second,
            min = argmin(signal).second;
            for (int n = 0; n < N; n++)
                pams[0][n] = ((max - min) * n) / (2.0 * (N - 1)) + min + (max - min)/4;

            set_params(pams);
        }
    }

    set_initial_hidden();
    if (seed <= 0){
        int add = H;
        for (int h = 0; h < H; ++h){
            set_initial_matrix(h, add-h);
        }
    }else{
        srand(seed);
        for (int h = 0; h < H; ++h){
            set_initial_random_matrix(h);
        }
    }
    if (get_log_em()) {
            cout << "INITIAL PARAMETERS" << endl;
            print(get_params());
            cout << "MATRIU DE TRANSICIÓ INICIAL" << endl;
            print(get_hidden_mat());
            for(int h = 0; h < H; ++h){
                cout << "Matriu inicial " << h << endl;
                print(get_trans_mat(h));
            }
        }
    
    double eps = tol + 1;
    for (int it=0; abs(eps) > tol and it < maxits; ++it){
        
        matrix_d old_params = get_params();
        matrix_d old_mat = get_hidden_mat();
        tensor_d old_mats = get_trans_mat();
        fit_model_params(signal, old_mat, old_mats);
        if (get_log_em()) {
            cout << "Iteration: " << it << endl;
            cout << "PARAMETERS" << endl;
            print(get_params());
            cout << "MATRIU DE TRANSICIÓ" << endl;
            print(get_hidden_mat());
            for(int h = 0; h < H; ++h){
                cout << "Matriu " << h << endl;
                print(get_trans_mat(h));
            }
        }
        
        eps = norm(get_params() - old_params) + norm(get_hidden_mat() - old_mat) + norm(get_trans_mat()-old_mats);
        
        if (it % iter_print == 0){
            cout << eps << endl;
        }
        
    }

    cout << "PARAMETERS FINALS" << endl;
    print(get_params());
    cout << "MATRIU DE TRANSICIÓ" << endl;
    print(get_hidden_mat());
    cout << "MATRIUS DE TRANSICIÓ" << endl;
    for(int h = 0; h < H; ++h){
        cout << "Matriu " << h << endl;
        print(get_trans_mat(h));
    }
    
}
*/
void HiddenHMM::EM_maximization(const vector_d &signal, const vector_int &states,
                                double tol, int maxits, int seed, int iter_print){

    int N = get_nstates(),
        H = get_nhidden();

    if (init_params){
        if (get_pdf_constant()){
            assert(signal.size() == states.size());
            gaussian_params(signal, states);
        }
        else{
            matrix_d pams(2, vector_d(N, 1));
            double max = argmax(signal).second,
            min = argmin(signal).second;
            for (int n = 0; n < N; n++)
                pams[0][n] = ((max - min) * n) / (2.0 * (N - 1)) + min + (max - min)/4;

            set_params(pams);
        }
    }

    set_initial_hidden();
    if (seed <= 0){
        int add = H;
        for (int h = 0; h < H; ++h){
            set_initial_matrix(h, add-h);
        }
    } else {
        srand(seed);
        for (int h = 0; h < H; ++h){
            set_initial_random_matrix(h);
        }
    }
    if (get_log_em()) {
            cout << "INITIAL PARAMETERS" << endl;
            print(get_params());
            cout << "MATRIU DE TRANSICIÓ INICIAL" << endl;
            cout << "HIDDEN GLOBALS:" << endl;
            cout << "Matriu Hidden inicial " << endl;
            print(get_hidden_mat());
            
            cout << "CHANNELS:" << endl;
            for(int h = 0; h < H; ++h){
                cout << "Matriu inicial " << h << endl;
                print(get_trans_mat(h));
            }
        }

    
    double eps = tol + 1;
    for (int it=0; abs(eps) > tol and it < maxits; ++it){
        
        matrix_d old_params = get_params();
        matrix_d old_hidden = get_hidden_mat();
        tensor_d old_mats = get_trans_mat();
        if (not state_training)
            fit_model_params(signal, old_hidden, old_mats);
        else
            fit_model_params_states(signal, states, old_hidden, old_mats);

        if (get_log_em()) {
            cout << "Iteration: " << it << endl;
            cout << "PARAMETERS" << endl;
            print(get_params());
            cout << "MATRIU DE TRANSICIÓ" << endl;
            cout << "HIDDEN GLOBALS:" << endl;
            cout << "Matriu Hidden " << endl;
            print(get_hidden_mat());
            cout << "CHANNELS:" << endl;
            for(int h = 0; h < H; ++h){
                cout << "Matriu " << h << endl;
                print(get_trans_mat(h));
            }
        }
        
        eps = norm(get_params() - old_params) + norm(get_hidden_mat() - old_hidden) + norm(get_trans_mat()-old_mats);
        
        if (it % iter_print == 0){
            cout << eps << endl;
        }
        
    }

    cout << "PARAMETERS FINALS" << endl;
    print(get_params());
    cout << "MATRIU DE TRANSICIÓ" << endl;
    cout << "HIDDEN GLOBALS:" << endl;
    cout << "Matriu Hidden " <<  endl;
    print(get_hidden_mat());
    cout << "CHANNELS:" << endl;
    cout << "MATRIUS DE TRANSICIÓ" << endl;
    for(int h = 0; h < H; ++h){
        cout << "Matriu " << h << endl;
        print(get_trans_mat(h));
    }
    
}
