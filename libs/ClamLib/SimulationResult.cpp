#include "SimulationResult.h"
#include "SimulationInfoBlockVector.h"

#include "../NumtoolsLib/NumtoolsLib.h"

#include <TGraph.h>
#include <sstream>

using namespace ClamLib;

SimulationResult::SimulationResult(TFile& file):
_file_ref(file),
_vec_sub(InitializeSubNetworkVector())
{
}

vector<DynamicSubNetwork> SimulationResult::InitializeSubNetworkVector() const
{
	vector<DynamicSubNetwork> vec_ret;

	SimulationInfoBlockVector* p_vec = static_cast<SimulationInfoBlockVector*>(_file_ref.Get("simulationinfoblockcollection"));
	vector<SimulationInfoBlock> vec_block = p_vec->BlockVector();

	for (
			vector<SimulationInfoBlock>::iterator iter = vec_block.begin();
			iter != vec_block.end();
			iter++
		)
			vec_ret.push_back(DynamicSubNetwork(*iter));

	return vec_ret;
}


//!
// \brief Return the rate for a given node id at the given time
// TODO: The interpolation type could be a parameter, with a default
// TODO: Ugly hack required as X values are not monotonically increasing
// TODO: InterpValue() will throw a numtoolsexception on out of bounds condition.
// It seems sensible to let this propagate to callers as there's no sensible return value to set.
//
DynamicLib::Rate SimulationResult::RateForIdByTime(Id id, DynamicLib::Time time) const
{
	if ( id._id_value <= 0)
		throw NumtoolsLib::NumtoolsException("Id values < 0 are invalid");

	std::stringstream ss(std::stringstream::in | std::stringstream::out);
	ss << "rate_" << id._id_value;

	TGraph *g;
	_file_ref.GetObject(ss.str().c_str(), g);
	if (!g)
		throw NumtoolsLib::NumtoolsException("Id  not found");

	const unsigned int SIZE = g->GetN();
	std::valarray<double> ys(g->GetY(), SIZE);
	std::valarray<double> xs(g->GetX(), SIZE);

	// URGH! Values in file are not monotonically increasing,
	// so just make SIZE values between 0..1
	const double INC = 1.0/SIZE;
	for (unsigned int i = 0; i < SIZE; ++i)
		xs[i] = i*INC;

	NumtoolsLib::Interpolator interp(NumtoolsLib::INTERP_AKIMA, xs, ys);
	const double result = interp.InterpValue(time);

	return result;
}
