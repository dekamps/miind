// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifndef _CODE_LIBS_CLAMLIB_HOMEOSTATICSMOOTH_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_HOMEOSTATICSMOOTH_INCLUDE_GUARD

#include "TrainedNet.h"

namespace ClamLib {
	//! This is a procedure which adapts the weights of a network trained with a HebbianAlgorithm. It is the user's
	//! responsibility to check that the network is really trained with a HebbianAlgorithm or that the procedure is
	//! applied appropriotely otherwise.
	//!
	//! Problem with a standard Hebbian network is that the overall constant is undetermined. In a multilayered network
	//! this constant can be taken one, but if the activities in the network are very small, as is often the case, this means that
	//! the weights are quadratically smaller. This means that activity that is propagated through the network is attentuated
	//! by each layer it passes. For example, if the typical level of activity of the training patterns is 0.1, the weights
	//! are of the order 0.01, and this is the maginitude of attenuation per layer. With four layers, the overall attenuation
	//! is 0.01^4 or 10^-8, which makes no sense. The input patterns of the TrainedNet are presented, and the networks are evolved.
	//! the weights are adjusted such that the average level of all active neurons is half of that of the input nodes.
	void HomeostaticSmooth(TrainedNet* p_net, double scale);
}

#endif // include guard