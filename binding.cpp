#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "calculation.cpp"

PYBIND11_MODULE(cppstats, m){
    m.doc("Correlation test calculations in C++");
    
    //calling pearson functions from python 
    m.def("pearson", &pearson_correlation_coefficient);
    m.def("pearson_p", &p_value_pearson); 
    
    //calling spearman functions from python 
    m.def("spearman", &spearman_rank_correlation);
    m.def("spearman_p", &p_value_spearman);
    
    //calling kendalltau 
    m.def("kendalltau", &kendall_tau); 
    m.def("kendalltau_p", &p_value_kendalltau); 

    //calling distance correlation from python 
    m.def("dCor", &distance_correlation); 
    m.def("dCor_p", &p_value_distance); 
}