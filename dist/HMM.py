# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_HMM')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_HMM')
    _HMM = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_HMM', [dirname(__file__)])
        except ImportError:
            import _HMM
            return _HMM
        try:
            _mod = imp.load_module('_HMM', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _HMM = swig_import_helper()
    del swig_import_helper
else:
    import _HMM
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class HMM(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, HMM, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, HMM, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _HMM.new_HMM(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def get_nstates(self):
        return _HMM.HMM_get_nstates(self)

    def get_params(self):
        return _HMM.HMM_get_params(self)

    def get_trans_mat(self):
        return _HMM.HMM_get_trans_mat(self)

    def get_initial_dist(self):
        return _HMM.HMM_get_initial_dist(self)

    def get_decode_iter_print(self):
        return _HMM.HMM_get_decode_iter_print(self)

    def get_print_decode(self):
        return _HMM.HMM_get_print_decode(self)

    def get_pdf_constant(self):
        return _HMM.HMM_get_pdf_constant(self)

    def get_log_em(self):
        return _HMM.HMM_get_log_em(self)

    def set_nstates(self, s):
        return _HMM.HMM_set_nstates(self, s)

    def set_params(self, mat):
        return _HMM.HMM_set_params(self, mat)

    def set_trans_mat(self, mat):
        return _HMM.HMM_set_trans_mat(self, mat)

    def set_initial_matrix(self):
        return _HMM.HMM_set_initial_matrix(self)

    def set_initial_random_matrix(self):
        return _HMM.HMM_set_initial_random_matrix(self)

    def set_initial_dist(self, v):
        return _HMM.HMM_set_initial_dist(self, v)

    def set_initial_means(self, means):
        return _HMM.HMM_set_initial_means(self, means)

    def set_initial_vars(self, means):
        return _HMM.HMM_set_initial_vars(self, means)

    def set_decode_iter_print(self, iter):
        return _HMM.HMM_set_decode_iter_print(self, iter)

    def set_print_decode(self, p):
        return _HMM.HMM_set_print_decode(self, p)

    def switch_log_em(self):
        return _HMM.HMM_switch_log_em(self)

    def switch_pdf_constant(self):
        return _HMM.HMM_switch_pdf_constant(self)

    def likelihood(self, signal):
        return _HMM.HMM_likelihood(self, signal)

    def fit_model_params_fast(self, signal):
        return _HMM.HMM_fit_model_params_fast(self, signal)

    def fit_model_params_from_truth(self, signal, v_states):
        return _HMM.HMM_fit_model_params_from_truth(self, signal, v_states)

    def fit_model_params(self, *args):
        return _HMM.HMM_fit_model_params(self, *args)

    def decode(self, signal, only_seq=True):
        return _HMM.HMM_decode(self, signal, only_seq)
    if _newclass:
        decode_seq = staticmethod(_HMM.HMM_decode_seq)
    else:
        decode_seq = _HMM.HMM_decode_seq
    if _newclass:
        decode_prob = staticmethod(_HMM.HMM_decode_prob)
    else:
        decode_prob = _HMM.HMM_decode_prob

    def EM_maximization(self, *args):
        return _HMM.HMM_EM_maximization(self, *args)
    __swig_destroy__ = _HMM.delete_HMM
    __del__ = lambda self: None
HMM_swigregister = _HMM.HMM_swigregister
HMM_swigregister(HMM)

def HMM_decode_seq(states, matrix, initial_dist, params, distr, prints=False, iter_print=10000):
    return _HMM.HMM_decode_seq(states, matrix, initial_dist, params, distr, prints, iter_print)
HMM_decode_seq = _HMM.HMM_decode_seq

def HMM_decode_prob(states, matrix, initial_dist, params, distr, prints=False, iter_print=10000):
    return _HMM.HMM_decode_prob(states, matrix, initial_dist, params, distr, prints, iter_print)
HMM_decode_prob = _HMM.HMM_decode_prob

class HiddenHMM(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, HiddenHMM, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, HiddenHMM, name)
    __repr__ = _swig_repr

    def __init__(self, s, h):
        this = _HMM.new_HiddenHMM(s, h)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def get_nstates(self):
        return _HMM.HiddenHMM_get_nstates(self)

    def get_nhidden(self):
        return _HMM.HiddenHMM_get_nhidden(self)

    def get_params(self):
        return _HMM.HiddenHMM_get_params(self)

    def get_trans_mat(self, *args):
        return _HMM.HiddenHMM_get_trans_mat(self, *args)

    def get_hidden_mat(self):
        return _HMM.HiddenHMM_get_hidden_mat(self)

    def get_initial_dist(self, *args):
        return _HMM.HiddenHMM_get_initial_dist(self, *args)

    def get_decode_iter_print(self):
        return _HMM.HiddenHMM_get_decode_iter_print(self)

    def get_print_decode(self):
        return _HMM.HiddenHMM_get_print_decode(self)

    def get_pdf_constant(self):
        return _HMM.HiddenHMM_get_pdf_constant(self)

    def get_log_em(self):
        return _HMM.HiddenHMM_get_log_em(self)

    def set_nstates(self, s):
        return _HMM.HiddenHMM_set_nstates(self, s)

    def set_nhidden(self, h):
        return _HMM.HiddenHMM_set_nhidden(self, h)

    def set_params(self, mat):
        return _HMM.HiddenHMM_set_params(self, mat)

    def set_trans_mat(self, *args):
        return _HMM.HiddenHMM_set_trans_mat(self, *args)

    def set_hidden_mat(self, mat):
        return _HMM.HiddenHMM_set_hidden_mat(self, mat)

    def set_initial_matrix(self, h, add):
        return _HMM.HiddenHMM_set_initial_matrix(self, h, add)

    def set_initial_random_matrix(self, h):
        return _HMM.HiddenHMM_set_initial_random_matrix(self, h)

    def set_initial_hidden(self):
        return _HMM.HiddenHMM_set_initial_hidden(self)

    def set_initial_dist(self, *args):
        return _HMM.HiddenHMM_set_initial_dist(self, *args)

    def set_initial_means(self, means):
        return _HMM.HiddenHMM_set_initial_means(self, means)

    def set_initial_vars(self, means):
        return _HMM.HiddenHMM_set_initial_vars(self, means)

    def set_decode_iter_print(self, iter):
        return _HMM.HiddenHMM_set_decode_iter_print(self, iter)

    def set_print_decode(self, p):
        return _HMM.HiddenHMM_set_print_decode(self, p)

    def switch_log_em(self):
        return _HMM.HiddenHMM_switch_log_em(self)

    def switch_init(self):
        return _HMM.HiddenHMM_switch_init(self)

    def switch_pdf_constant(self):
        return _HMM.HiddenHMM_switch_pdf_constant(self)

    def switch_state_training(self):
        return _HMM.HiddenHMM_switch_state_training(self)

    def fit_model_params(self, signal, hidden_mat, trans_mats):
        return _HMM.HiddenHMM_fit_model_params(self, signal, hidden_mat, trans_mats)

    def fit_model_params_from_truth(self, signal, v_states):
        return _HMM.HiddenHMM_fit_model_params_from_truth(self, signal, v_states)

    def fit_model_params_states(self, signal, states, hidden_mat, trans_mats):
        return _HMM.HiddenHMM_fit_model_params_states(self, signal, states, hidden_mat, trans_mats)

    def decode(self, signal, only_seq=True):
        return _HMM.HiddenHMM_decode(self, signal, only_seq)

    def decode_probs(self, signal):
        return _HMM.HiddenHMM_decode_probs(self, signal)
    if _newclass:
        decode_seq = staticmethod(_HMM.HiddenHMM_decode_seq)
    else:
        decode_seq = _HMM.HiddenHMM_decode_seq

    def decode_seq_probs(self, S, H, hidden_mat, matrices, initial_distr, params, distr, prints, iter_print):
        return _HMM.HiddenHMM_decode_seq_probs(self, S, H, hidden_mat, matrices, initial_distr, params, distr, prints, iter_print)

    def EM_maximization(self, *args):
        return _HMM.HiddenHMM_EM_maximization(self, *args)
    __swig_destroy__ = _HMM.delete_HiddenHMM
    __del__ = lambda self: None
HiddenHMM_swigregister = _HMM.HiddenHMM_swigregister
HiddenHMM_swigregister(HiddenHMM)

def HiddenHMM_decode_seq(S, H, hidden_mat, matrices, initial_distr, params, distr, prints, iter_print=10000):
    return _HMM.HiddenHMM_decode_seq(S, H, hidden_mat, matrices, initial_distr, params, distr, prints, iter_print)
HiddenHMM_decode_seq = _HMM.HiddenHMM_decode_seq

class AdaptiveHHMM(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, AdaptiveHHMM, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, AdaptiveHHMM, name)
    __repr__ = _swig_repr

    def __init__(self, s, h):
        this = _HMM.new_AdaptiveHHMM(s, h)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def get_nstates(self):
        return _HMM.AdaptiveHHMM_get_nstates(self)

    def get_nhidden(self):
        return _HMM.AdaptiveHHMM_get_nhidden(self)

    def get_params(self):
        return _HMM.AdaptiveHHMM_get_params(self)

    def get_trans_mat(self, *args):
        return _HMM.AdaptiveHHMM_get_trans_mat(self, *args)

    def get_hidden_mat(self, *args):
        return _HMM.AdaptiveHHMM_get_hidden_mat(self, *args)

    def get_initial_dist(self, *args):
        return _HMM.AdaptiveHHMM_get_initial_dist(self, *args)

    def get_decode_iter_print(self):
        return _HMM.AdaptiveHHMM_get_decode_iter_print(self)

    def get_print_decode(self):
        return _HMM.AdaptiveHHMM_get_print_decode(self)

    def get_pdf_constant(self):
        return _HMM.AdaptiveHHMM_get_pdf_constant(self)

    def get_log_em(self):
        return _HMM.AdaptiveHHMM_get_log_em(self)

    def get_state_training(self):
        return _HMM.AdaptiveHHMM_get_state_training(self)

    def set_nstates(self, s):
        return _HMM.AdaptiveHHMM_set_nstates(self, s)

    def set_nhidden(self, h):
        return _HMM.AdaptiveHHMM_set_nhidden(self, h)

    def set_params(self, mat):
        return _HMM.AdaptiveHHMM_set_params(self, mat)

    def set_trans_mat(self, *args):
        return _HMM.AdaptiveHHMM_set_trans_mat(self, *args)

    def set_hidden_mat(self, *args):
        return _HMM.AdaptiveHHMM_set_hidden_mat(self, *args)

    def set_initial_matrix(self, h, add):
        return _HMM.AdaptiveHHMM_set_initial_matrix(self, h, add)

    def set_initial_random_matrix(self, h):
        return _HMM.AdaptiveHHMM_set_initial_random_matrix(self, h)

    def set_initial_hidden(self):
        return _HMM.AdaptiveHHMM_set_initial_hidden(self)

    def set_initial_dist(self, *args):
        return _HMM.AdaptiveHHMM_set_initial_dist(self, *args)

    def set_initial_means(self, means):
        return _HMM.AdaptiveHHMM_set_initial_means(self, means)

    def set_initial_vars(self, means):
        return _HMM.AdaptiveHHMM_set_initial_vars(self, means)

    def set_decode_iter_print(self, iter):
        return _HMM.AdaptiveHHMM_set_decode_iter_print(self, iter)

    def set_print_decode(self, p):
        return _HMM.AdaptiveHHMM_set_print_decode(self, p)

    def switch_log_em(self):
        return _HMM.AdaptiveHHMM_switch_log_em(self)

    def switch_init(self):
        return _HMM.AdaptiveHHMM_switch_init(self)

    def switch_pdf_constant(self):
        return _HMM.AdaptiveHHMM_switch_pdf_constant(self)

    def switch_state_training(self):
        return _HMM.AdaptiveHHMM_switch_state_training(self)

    def fit_model_params(self, signal, hidden_mat, trans_mats):
        return _HMM.AdaptiveHHMM_fit_model_params(self, signal, hidden_mat, trans_mats)

    def fit_model_params_states(self, signal, states, hidden_mat, trans_mats):
        return _HMM.AdaptiveHHMM_fit_model_params_states(self, signal, states, hidden_mat, trans_mats)

    def fit_model_params_from_truth(self, signal, v_states):
        return _HMM.AdaptiveHHMM_fit_model_params_from_truth(self, signal, v_states)

    def decode(self, signal, only_seq=True):
        return _HMM.AdaptiveHHMM_decode(self, signal, only_seq)
    if _newclass:
        decode_seq = staticmethod(_HMM.AdaptiveHHMM_decode_seq)
    else:
        decode_seq = _HMM.AdaptiveHHMM_decode_seq

    def EM_maximization(self, *args):
        return _HMM.AdaptiveHHMM_EM_maximization(self, *args)
    __swig_destroy__ = _HMM.delete_AdaptiveHHMM
    __del__ = lambda self: None
AdaptiveHHMM_swigregister = _HMM.AdaptiveHHMM_swigregister
AdaptiveHHMM_swigregister(AdaptiveHHMM)

def AdaptiveHHMM_decode_seq(S, H, hidden_mat, matrices, initial_distr, params, distr, prints, iter_print=10000):
    return _HMM.AdaptiveHHMM_decode_seq(S, H, hidden_mat, matrices, initial_distr, params, distr, prints, iter_print)
AdaptiveHHMM_decode_seq = _HMM.AdaptiveHHMM_decode_seq


def numpy2vec(numpy_array):
    return _HMM.numpy2vec(numpy_array)
numpy2vec = _HMM.numpy2vec

def array2vec(arr, size):
    return _HMM.array2vec(arr, size)
array2vec = _HMM.array2vec

def array2vec_int(arr, size):
    return _HMM.array2vec_int(arr, size)
array2vec_int = _HMM.array2vec_int

def numpy2vec_int(numpy_array):
    return _HMM.numpy2vec_int(numpy_array)
numpy2vec_int = _HMM.numpy2vec_int
class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _HMM.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return _HMM.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _HMM.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _HMM.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _HMM.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _HMM.SwigPyIterator_equal(self, x)

    def copy(self):
        return _HMM.SwigPyIterator_copy(self)

    def next(self):
        return _HMM.SwigPyIterator_next(self)

    def __next__(self):
        return _HMM.SwigPyIterator___next__(self)

    def previous(self):
        return _HMM.SwigPyIterator_previous(self)

    def advance(self, n):
        return _HMM.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _HMM.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _HMM.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _HMM.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _HMM.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _HMM.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _HMM.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _HMM.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class tensord(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, tensord, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, tensord, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _HMM.tensord_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _HMM.tensord___nonzero__(self)

    def __bool__(self):
        return _HMM.tensord___bool__(self)

    def __len__(self):
        return _HMM.tensord___len__(self)

    def __getslice__(self, i, j):
        return _HMM.tensord___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _HMM.tensord___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _HMM.tensord___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _HMM.tensord___delitem__(self, *args)

    def __getitem__(self, *args):
        return _HMM.tensord___getitem__(self, *args)

    def __setitem__(self, *args):
        return _HMM.tensord___setitem__(self, *args)

    def pop(self):
        return _HMM.tensord_pop(self)

    def append(self, x):
        return _HMM.tensord_append(self, x)

    def empty(self):
        return _HMM.tensord_empty(self)

    def size(self):
        return _HMM.tensord_size(self)

    def swap(self, v):
        return _HMM.tensord_swap(self, v)

    def begin(self):
        return _HMM.tensord_begin(self)

    def end(self):
        return _HMM.tensord_end(self)

    def rbegin(self):
        return _HMM.tensord_rbegin(self)

    def rend(self):
        return _HMM.tensord_rend(self)

    def clear(self):
        return _HMM.tensord_clear(self)

    def get_allocator(self):
        return _HMM.tensord_get_allocator(self)

    def pop_back(self):
        return _HMM.tensord_pop_back(self)

    def erase(self, *args):
        return _HMM.tensord_erase(self, *args)

    def __init__(self, *args):
        this = _HMM.new_tensord(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _HMM.tensord_push_back(self, x)

    def front(self):
        return _HMM.tensord_front(self)

    def back(self):
        return _HMM.tensord_back(self)

    def assign(self, n, x):
        return _HMM.tensord_assign(self, n, x)

    def resize(self, *args):
        return _HMM.tensord_resize(self, *args)

    def insert(self, *args):
        return _HMM.tensord_insert(self, *args)

    def reserve(self, n):
        return _HMM.tensord_reserve(self, n)

    def capacity(self):
        return _HMM.tensord_capacity(self)
    __swig_destroy__ = _HMM.delete_tensord
    __del__ = lambda self: None
tensord_swigregister = _HMM.tensord_swigregister
tensord_swigregister(tensord)

class vectord(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, vectord, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, vectord, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _HMM.vectord_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _HMM.vectord___nonzero__(self)

    def __bool__(self):
        return _HMM.vectord___bool__(self)

    def __len__(self):
        return _HMM.vectord___len__(self)

    def __getslice__(self, i, j):
        return _HMM.vectord___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _HMM.vectord___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _HMM.vectord___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _HMM.vectord___delitem__(self, *args)

    def __getitem__(self, *args):
        return _HMM.vectord___getitem__(self, *args)

    def __setitem__(self, *args):
        return _HMM.vectord___setitem__(self, *args)

    def pop(self):
        return _HMM.vectord_pop(self)

    def append(self, x):
        return _HMM.vectord_append(self, x)

    def empty(self):
        return _HMM.vectord_empty(self)

    def size(self):
        return _HMM.vectord_size(self)

    def swap(self, v):
        return _HMM.vectord_swap(self, v)

    def begin(self):
        return _HMM.vectord_begin(self)

    def end(self):
        return _HMM.vectord_end(self)

    def rbegin(self):
        return _HMM.vectord_rbegin(self)

    def rend(self):
        return _HMM.vectord_rend(self)

    def clear(self):
        return _HMM.vectord_clear(self)

    def get_allocator(self):
        return _HMM.vectord_get_allocator(self)

    def pop_back(self):
        return _HMM.vectord_pop_back(self)

    def erase(self, *args):
        return _HMM.vectord_erase(self, *args)

    def __init__(self, *args):
        this = _HMM.new_vectord(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _HMM.vectord_push_back(self, x)

    def front(self):
        return _HMM.vectord_front(self)

    def back(self):
        return _HMM.vectord_back(self)

    def assign(self, n, x):
        return _HMM.vectord_assign(self, n, x)

    def resize(self, *args):
        return _HMM.vectord_resize(self, *args)

    def insert(self, *args):
        return _HMM.vectord_insert(self, *args)

    def reserve(self, n):
        return _HMM.vectord_reserve(self, n)

    def capacity(self):
        return _HMM.vectord_capacity(self)
    __swig_destroy__ = _HMM.delete_vectord
    __del__ = lambda self: None
vectord_swigregister = _HMM.vectord_swigregister
vectord_swigregister(vectord)

class vectori(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, vectori, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, vectori, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _HMM.vectori_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _HMM.vectori___nonzero__(self)

    def __bool__(self):
        return _HMM.vectori___bool__(self)

    def __len__(self):
        return _HMM.vectori___len__(self)

    def __getslice__(self, i, j):
        return _HMM.vectori___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _HMM.vectori___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _HMM.vectori___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _HMM.vectori___delitem__(self, *args)

    def __getitem__(self, *args):
        return _HMM.vectori___getitem__(self, *args)

    def __setitem__(self, *args):
        return _HMM.vectori___setitem__(self, *args)

    def pop(self):
        return _HMM.vectori_pop(self)

    def append(self, x):
        return _HMM.vectori_append(self, x)

    def empty(self):
        return _HMM.vectori_empty(self)

    def size(self):
        return _HMM.vectori_size(self)

    def swap(self, v):
        return _HMM.vectori_swap(self, v)

    def begin(self):
        return _HMM.vectori_begin(self)

    def end(self):
        return _HMM.vectori_end(self)

    def rbegin(self):
        return _HMM.vectori_rbegin(self)

    def rend(self):
        return _HMM.vectori_rend(self)

    def clear(self):
        return _HMM.vectori_clear(self)

    def get_allocator(self):
        return _HMM.vectori_get_allocator(self)

    def pop_back(self):
        return _HMM.vectori_pop_back(self)

    def erase(self, *args):
        return _HMM.vectori_erase(self, *args)

    def __init__(self, *args):
        this = _HMM.new_vectori(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _HMM.vectori_push_back(self, x)

    def front(self):
        return _HMM.vectori_front(self)

    def back(self):
        return _HMM.vectori_back(self)

    def assign(self, n, x):
        return _HMM.vectori_assign(self, n, x)

    def resize(self, *args):
        return _HMM.vectori_resize(self, *args)

    def insert(self, *args):
        return _HMM.vectori_insert(self, *args)

    def reserve(self, n):
        return _HMM.vectori_reserve(self, n)

    def capacity(self):
        return _HMM.vectori_capacity(self)
    __swig_destroy__ = _HMM.delete_vectori
    __del__ = lambda self: None
vectori_swigregister = _HMM.vectori_swigregister
vectori_swigregister(vectori)

class matrixd(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, matrixd, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, matrixd, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _HMM.matrixd_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _HMM.matrixd___nonzero__(self)

    def __bool__(self):
        return _HMM.matrixd___bool__(self)

    def __len__(self):
        return _HMM.matrixd___len__(self)

    def __getslice__(self, i, j):
        return _HMM.matrixd___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _HMM.matrixd___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _HMM.matrixd___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _HMM.matrixd___delitem__(self, *args)

    def __getitem__(self, *args):
        return _HMM.matrixd___getitem__(self, *args)

    def __setitem__(self, *args):
        return _HMM.matrixd___setitem__(self, *args)

    def pop(self):
        return _HMM.matrixd_pop(self)

    def append(self, x):
        return _HMM.matrixd_append(self, x)

    def empty(self):
        return _HMM.matrixd_empty(self)

    def size(self):
        return _HMM.matrixd_size(self)

    def swap(self, v):
        return _HMM.matrixd_swap(self, v)

    def begin(self):
        return _HMM.matrixd_begin(self)

    def end(self):
        return _HMM.matrixd_end(self)

    def rbegin(self):
        return _HMM.matrixd_rbegin(self)

    def rend(self):
        return _HMM.matrixd_rend(self)

    def clear(self):
        return _HMM.matrixd_clear(self)

    def get_allocator(self):
        return _HMM.matrixd_get_allocator(self)

    def pop_back(self):
        return _HMM.matrixd_pop_back(self)

    def erase(self, *args):
        return _HMM.matrixd_erase(self, *args)

    def __init__(self, *args):
        this = _HMM.new_matrixd(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _HMM.matrixd_push_back(self, x)

    def front(self):
        return _HMM.matrixd_front(self)

    def back(self):
        return _HMM.matrixd_back(self)

    def assign(self, n, x):
        return _HMM.matrixd_assign(self, n, x)

    def resize(self, *args):
        return _HMM.matrixd_resize(self, *args)

    def insert(self, *args):
        return _HMM.matrixd_insert(self, *args)

    def reserve(self, n):
        return _HMM.matrixd_reserve(self, n)

    def capacity(self):
        return _HMM.matrixd_capacity(self)
    __swig_destroy__ = _HMM.delete_matrixd
    __del__ = lambda self: None
matrixd_swigregister = _HMM.matrixd_swigregister
matrixd_swigregister(matrixd)

class matrixi(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, matrixi, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, matrixi, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _HMM.matrixi_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _HMM.matrixi___nonzero__(self)

    def __bool__(self):
        return _HMM.matrixi___bool__(self)

    def __len__(self):
        return _HMM.matrixi___len__(self)

    def __getslice__(self, i, j):
        return _HMM.matrixi___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _HMM.matrixi___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _HMM.matrixi___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _HMM.matrixi___delitem__(self, *args)

    def __getitem__(self, *args):
        return _HMM.matrixi___getitem__(self, *args)

    def __setitem__(self, *args):
        return _HMM.matrixi___setitem__(self, *args)

    def pop(self):
        return _HMM.matrixi_pop(self)

    def append(self, x):
        return _HMM.matrixi_append(self, x)

    def empty(self):
        return _HMM.matrixi_empty(self)

    def size(self):
        return _HMM.matrixi_size(self)

    def swap(self, v):
        return _HMM.matrixi_swap(self, v)

    def begin(self):
        return _HMM.matrixi_begin(self)

    def end(self):
        return _HMM.matrixi_end(self)

    def rbegin(self):
        return _HMM.matrixi_rbegin(self)

    def rend(self):
        return _HMM.matrixi_rend(self)

    def clear(self):
        return _HMM.matrixi_clear(self)

    def get_allocator(self):
        return _HMM.matrixi_get_allocator(self)

    def pop_back(self):
        return _HMM.matrixi_pop_back(self)

    def erase(self, *args):
        return _HMM.matrixi_erase(self, *args)

    def __init__(self, *args):
        this = _HMM.new_matrixi(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _HMM.matrixi_push_back(self, x)

    def front(self):
        return _HMM.matrixi_front(self)

    def back(self):
        return _HMM.matrixi_back(self)

    def assign(self, n, x):
        return _HMM.matrixi_assign(self, n, x)

    def resize(self, *args):
        return _HMM.matrixi_resize(self, *args)

    def insert(self, *args):
        return _HMM.matrixi_insert(self, *args)

    def reserve(self, n):
        return _HMM.matrixi_reserve(self, n)

    def capacity(self):
        return _HMM.matrixi_capacity(self)
    __swig_destroy__ = _HMM.delete_matrixi
    __del__ = lambda self: None
matrixi_swigregister = _HMM.matrixi_swigregister
matrixi_swigregister(matrixi)

# This file is compatible with both classic and new-style classes.

