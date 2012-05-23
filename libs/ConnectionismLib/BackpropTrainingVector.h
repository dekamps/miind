// Copyright (c) 2005 - 2009 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_LIBS_CONNECTIONISM_BACKPROPTRAININGVECTOR_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISM_BACKPROPTRAININGVECTOR_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <vector>
#include "../NetLib/NetLib.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "LayeredNetwork.h"

using std::vector;

using NumtoolsLib::UniformDistribution;


namespace ConnectionismLib {

	enum TrainingMode {BATCH, RANDOM};

	//! BackpropTrainingVector
	template <class LayeredImplementation, class Iterator>
	class BackpropTrainingVector {
    public:

    typedef LayeredNetwork<LayeredImplementation>*	network_pointer;
	typedef TrainingUnit<typename LayeredImplementation::NodeValue> TU;

    BackpropTrainingVector
    ( 
		LayeredNetwork<LayeredImplementation>* ,  
		const TrainingParameter&, 
		TrainingMode = RANDOM 
	);
 
	//! virtual destructor
	virtual ~BackpropTrainingVector();

    bool CreateCategorization    
	(
		const vector<D_Pattern*>& 
	);

    bool AcceptTrainingUnitVector
	(
		const vector<TU>&
	);

    bool PushBack                
	(
		const TrainingUnit<typename LayeredImplementation::NodeValue>&
	);

    const TU& operator[]
	(
		size_t 
	) const;	

    size_t Size() const {return _vec_tu.size(); }

	// training related methods:
	void	Train();
	double	ErrorValue() const;

    private:

		double ErrorValue(const D_Pattern&, const D_Pattern&) const;

		const TrainingParameter	_parameter_train;
		TrainingMode			_mode;   // batch or random mode
		vector<TU>				_vec_tu; // pointer to training units
		bool					_lock;   // lock vector after categorization
								 // no more TU's should be added
    
		LayeredNetwork<LayeredImplementation>*	_p_net;	 // pointer to network to train

	}; // end of BackpropTrainingVector

} // end of Connectionism

#endif // include guard
