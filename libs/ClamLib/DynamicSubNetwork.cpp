#include "DynamicSubNetwork.h"

using namespace ClamLib;

DynamicSubNetwork::DynamicSubNetwork()
// _block has a default ctor
{
}

DynamicSubNetwork::DynamicSubNetwork(const SimulationInfoBlock& block):
_block(block)
{
}

string DynamicSubNetwork::GetName() const
{
	return string(_block.GetName());
}

DynamicSubNetwork::const_iterator DynamicSubNetwork::begin() const
{
	DynamicSubNetwork::const_iterator iter(_block.InfoVector().begin()+1,_block.InfoVector().end(),_block.DescriptionVector());
	return iter;
}

DynamicSubNetwork::const_iterator DynamicSubNetwork::end() const
{
	return this->begin(_block.DescriptionVector().NumberOfLayers());
}

DynamicSubNetwork::const_iterator DynamicSubNetwork::begin(Layer l) const
{
	// the case l == NumberOfLayers() corresponds to the end() of the network and is therefore allowed
	// derefencing is caught in the ForwardSubNetworkIterator class

	assert(l <= static_cast<Layer>(_block.DescriptionVector().NumberOfLayers()) );
	int n_neurons = 0;
	for (Layer i = 0; i < l ; i++)
	{
		RootLayerDescription desc = _block.DescriptionVector()[i];
		n_neurons += desc._nr_x_pixels*desc._nr_y_pixels*desc._nr_features;
	}
	DynamicSubNetwork::const_iterator iter(_block.InfoVector().begin() + n_neurons + 1,_block.InfoVector().end(),_block.DescriptionVector());
	return iter;
}

DynamicSubNetwork::const_iterator DynamicSubNetwork::end(Layer l) const
{

	if (l == _block.DescriptionVector().NumberOfLayers() )
		throw ClamLibException(EXCEP);

	return this->begin(++l);
}

DynamicSubNetwork::const_rziterator DynamicSubNetwork::rzbegin() const
{
	return this->rzbegin(0);
}

DynamicSubNetwork::const_rziterator DynamicSubNetwork::rzend() const
{
	return this->rzbegin(_block.DescriptionVector().NumberOfLayers());
}

DynamicSubNetwork::const_rziterator DynamicSubNetwork::rzbegin(Layer l) const
{
	// the case l == NumberOfLayers() corresponds to the end() of the network and is therefore allowed
	// derefencing is caught in the ForwardSubNetworkIterator class
	assert(l <= static_cast<Layer>(_block.DescriptionVector().NumberOfLayers()) );

	// There are more circuits than those that correpond to the original ANN, for example
	// external inputs.
	// Therefore the end must be one beyond those CircuitInfo's that correspond to 
	// ANN nodes, but to the end of InfoVector
	
	Number n_layers = _block.DescriptionVector().NumberOfLayers();
	int n_neurons = 0;
	for (Layer i = 0; i < l; i++)
	{
		RootLayerDescription desc = _block.DescriptionVector()[n_layers-i-1];
		n_neurons += desc._nr_x_pixels*desc._nr_y_pixels*desc._nr_features;
	}
	DynamicSubNetwork::const_rziterator 
		iter
		(
			_block.InfoVector().begin()+n_neurons+1,
			_block.InfoVector(),
			_block.DescriptionVector()
		);

	return iter;
}

DynamicSubNetwork::const_rziterator DynamicSubNetwork::rzend(Layer l) const
{
	if (l == _block.DescriptionVector().NumberOfLayers() )
		throw ClamLibException(EXCEP);

	return this->rzbegin(++l);
}

void DynamicSubNetwork::Write() const
{
	_block.Write();
}

const CircuitDescription& DynamicSubNetwork::GetCircuitDescription() const
{
	return _block.DescriptionCircuit();
}

const string DynamicSubNetwork::EXCEP("Invalid SubNetworkIterator");
