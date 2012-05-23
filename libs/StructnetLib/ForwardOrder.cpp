#include "ForwardOrder.h"
#include "StructnetLibException.h"

using namespace StructnetLib;

ForwardOrder::ForwardOrder
(
	Index id,
	const vector<Pair>& vec
):
_id(id),
_vec_ref(vec)
{
}

NodeId ForwardOrder::Id() const
{
	if ( _id >= _vec_ref.size() )
		throw StructnetLibException("Try to call Id() beyond vector boundary");
	return _vec_ref[_id].first;
}

const PhysicalPosition& ForwardOrder::Position() const
{
	if (_id >= _vec_ref.size() )
		throw StructnetLibException("Try to call Id() beyond vector boundary");
	return _vec_ref[_id].second;
}