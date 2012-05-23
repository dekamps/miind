#include "ReverseSubNetworkIterator.h"

using namespace ClamLib;

ReverseSubNetworkIterator::ReverseSubNetworkIterator
(
	vector<CircuitInfo>::const_iterator iter, 
	const std::vector<CircuitInfo> &vec_circuit, 
	const RootLayeredNetDescription &desc
):
_iter(iter),
_vec_circuit(vec_circuit),
_id_pos(ToLayeredDescriptionVector(desc)),
_rz_order(_id_pos.rzbegin() + (static_cast<int>(iter - vec_circuit.begin()) - 1)),
_desc(desc)
// the -1 compensates for the fact that the first CircuitInfo corresponds to NodeId(0),
// which has no circuit associated with it.
{
			// the first element of the NodeIdPositionPair vector starts at 1
			// so determine where in the rz_order, unlesss iter == end()
			if (_iter != _vec_circuit.end())
				_iter = _vec_circuit.begin() +  _rz_order.Id()._id_value;
}


ReverseSubNetworkIterator::ReverseSubNetworkIterator(const ReverseSubNetworkIterator& iter):
_iter(iter._iter),
_vec_circuit(iter._vec_circuit),
_id_pos(iter._id_pos),
_rz_order(iter._rz_order),
_desc(iter._desc)
{
}

bool ReverseSubNetworkIterator::Position(PhysicalPosition* ppos) const
{
	if (_iter < _vec_circuit.end() )
		*ppos = _id_pos.Position(NodeId(_iter->IdOriginal()._id_value));
	else 
		throw ClamLibException(EXCEP);

	return true;
}

const string ReverseSubNetworkIterator::EXCEP("Invalid ReverseSubNetworkIterator");