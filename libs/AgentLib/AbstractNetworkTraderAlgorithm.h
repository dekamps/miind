#include "DynamicLib/AbstractAlgorithmCode.h"
#include "DynamicLib/DynamicLib.h"
#include <vector>
#include <iostream>

using namespace DynamicLib;
using namespace SparseImplementationLib;
using std::cout;
using std::endl;


#ifndef ABSTRACT_NTA
#define ABSTRACT_NTA
class AbstractNetworkTraderAlgorithm : public AbstractAlgorithm<double>
{
	public:
		AbstractNetworkTraderAlgorithm( void );
		~AbstractNetworkTraderAlgorithm( void );

		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time 
		);

		//! clone
		virtual AbstractNetworkTraderAlgorithm* Clone() const = 0;

		//! report initial state
		virtual AlgorithmGrid Grid() const;

		//! report node state
		virtual NodeState State() const;

		//!
		virtual string LogString() const = 0;

		//! streaming to output stream
		virtual bool ToStream(ostream&) const;

		//! streaming from input stream
		virtual bool FromStream(istream&);

		//! stream Tag
		string Tag() const;

		virtual bool Dump(ostream&) const;

		virtual bool Configure
		(
			const SimulationRunParameter&
		);

		virtual Time CurrentTime() const; 

		virtual Rate CurrentRate() const; 

	protected:
		
		vector<double> _state;
		
		int _time;

		bool _is_update_mode;

		double _new_price;

		int _id;

};
#endif