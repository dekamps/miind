%module Util
%{
	#include "SparseNode.h"
	#include "SparseImplementation.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_pair.i"

%template(vector_double) std::vector<double>;
%template(vector_string) std::vector<std::string>;
%template(vector_int) std::vector<int>;
%template(pairdd) std::pair<double, double>;


%include "SparseNode.h"
%include "SparseImplementation.h"

%template(D_SparseNode) SparseNode<double,double>;
%template(D_SparseImplementation) SparseImplementationLib<SparseNode<double, double > >;
