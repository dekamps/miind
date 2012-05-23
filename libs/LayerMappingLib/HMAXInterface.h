#ifndef LAYERMAPPINGLIB_HMAXINTERFACE_H
#define LAYERMAPPINGLIB_HMAXINTERFACE_H

#include <vector>

#include "NetworkInterfaceCode.h"
#include "NetworkEnsembleCode.h"
#include "FeatureMapNetworkCode.h"

using namespace std;

namespace LayerMappingLib
{
	class HMAXInterface : NetworkInterface<NetworkEnsemble<double> >
	{
		public:
		typedef NetworkEnsemble<double> Network;
		typedef double value_type;
	
		HMAXInterface( Network n );

		vector<value_type> feature_vector();

		void set_input( const std::vector<value_type>& image );
		void set_input( value_type* image );

		private:
	};
}
#endif //LAYERMAPPINGLIB_HMAXINTERFACE_H
