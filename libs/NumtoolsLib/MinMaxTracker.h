// Copyright (c) 2005 - 2009 Marc de Kamps, Korbinian Trumpp
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software 
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY 
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
// Coded by Korbo...
#ifndef _CODE_LIBS_NUMTOOLSLIB_MINMAXTRACKER_INCLUDE_GUARD
#define _CODE_LIBS_NUMTOOLSLIB_MINMAXTRACKER_INCLUDE_GUARD

#include <limits>

using std::numeric_limits;

namespace NumtoolsLib
{

	//!
	//! Class providing a tracking possibility for min and max of all
	//! values fed.
	
	template <class ValueType>
	class MinMaxTracker
	{
	public:

		/// Constructor.
		MinMaxTracker();

		/// Reset the tracker and its min and max.
		void reset();
	
		/// Feed one value into the tracker.
		void feedValue( ValueType );
	
		/// Feed one value into the tracker indicating the index of the value.
		void feedValue( ValueType, unsigned int );
	
		/// Return the min value of all values fed so far.
		ValueType getMin();
	
		/// Return the max value of all values fed so far.
		ValueType getMax();
	
		/// Return the index of min value element.
		unsigned int getMinIndex();
		
		/// Return the index of max value element.
		unsigned int getMaxIndex();
		
		/// Return the avg value of all values fed so far.
		ValueType getAvg();
		
		/// Return the avg value of all values fed so far.
		int nrOfFeeds();

	private:
		
		/// Internally stored minimum so far.
		ValueType _min;
		/// Internally stored maximum so far.
		ValueType _max;
		/// Internally stored average.
		ValueType _sum_for_avg;
		/// Internally stored average.
		int _nr_of_values;
		/// Index can be used for indicating which is min
		unsigned int _min_index;
		/// Index can be used for indicating which is max
		unsigned int _max_index;
		/// Internal object for value limits.
		numeric_limits<ValueType> _limits;

	}; // end of MinMaxTracker

} // end of namespace Numtools


#endif // include guard
