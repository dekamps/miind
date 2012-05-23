%module LayerMapping
%{
	#include "Models.h"
	#include "NetworkInterface.h"
	#include "NetworkInterfaceCode.h"
/* 	#include "HMAXInterface.h" */

/* 	#include "gtkmm/LayerMappingApplet.h" */
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_pair.i"

%template(vector_double) std::vector<double>;
%template(vector_string) std::vector<std::string>;
%template(vector_int) std::vector<int>;
%template(pairdd) std::pair<double, double>;


%include "NetworkEnsemble.h"
%include "NetworkEnsembleCode.h"

%include "NetworkInterface.h"
%include "NetworkInterfaceCode.h"

%include "FunctionFactory.h"

%template(NI_NetworkEnsemble) LayerMappingLib::NetworkInterface<LayerMappingLib::NetworkEnsemble<double> >;

/* %include "HMAXInterface.h" */

%include "Models.h"

/* %include "gtkmm/LayerMappingApplet.h" */