#include "ForwardSubNetworkIterator.h"

using namespace ClamLib;

bool ForwardSubNetworkIterator::Position(PhysicalPosition* ppos) const
{	
	if (_iter < _iter_end)
		*ppos = _id_pos.Position(NodeId(_iter->IdOriginal()._id_value));
	else
		throw ClamLibException(EXCEP);

	return true;
}

const string ForwardSubNetworkIterator::EXCEP("Invalid ForwardSubNetworkIterator");