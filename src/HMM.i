%module HMM
%{
#define SWIG_FILE_WITH_INIT
#include "HMM.hh"
#include "Numpy2vec/nptostdvect.hh" 
#include "HiddenHMM.hh"
#include "HiddenHMM_comp.hh"
%}

%include "HMM.hh"
%include "HiddenHMM.hh"
%include "HiddenHMM_comp.hh"
%include "Numpy2vec/nptostdvect.hh"
%include "std_vector.i"
%template(tensord) std::vector<std::vector<std::vector<double>>>;
%template(vectord) std::vector<double>;
%template(vectori) std::vector<int>;
%template(matrixd) std::vector<std::vector<double>>;
%template(matrixi) std::vector<std::vector<int>>;
