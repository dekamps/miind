#ifndef _CODE_LIBS_CLAMLIB_ROOTNETCATALOG_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_ROOTNETCATALOG_INCLUDE_GUARD

#include <TNamed.h>

class RootNetCatalog : public TNamed {
public:

	RootNetCatalog
	(
		Int_t,							/*!< Number of nodes in DynamicNetwork					*/
		const vector<RootMetaNet&>)&	/*!< vector of MetaNets, defineing the DynamicNetwork	*/
	);


	ClassDef(RootNetCatalog,1);
private:

	Int_t						_number_dynamic_nodes;
	const vector<RootMetaNet&>	_vector_networks;

};

#endif // include guard
