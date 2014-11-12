// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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
#ifndef MPILIB_NODETYPE_HPP_
#define MPILIB_NODETYPE_HPP_

namespace MPILib {

//! This determines an MPINode's type, which will be checked when Dale's law is set.
//
//! MPINetwork objects will check Dale's law by default, although this can be switched off. For neuroscience simulations, it has been shown
//! that uncertainty on whether inhibitory connections should be negatively valued can lead to bugs if Dale's law is not checked.. The convention
//! is that inhibitory weights should have a negative value.  
//! Ultimately, an Algorithm must do a conversion from its external input contributions to internal parameters that  how to evolve its internal state. So,
//! it is the responsibility of the developer of the Algorithm to document how it will use this information.
enum NodeType {
	NEUTRAL,		//! Indicates that Dale's law should not be checked for this node 
	EXCITATORY_GAUSSIAN,    //! Indicates that Dale's law should be checked and that the contribution oof this excitatory node is to be interpreted as additive Guassian noise	
	INHIBITORY_GAUSSIAN, 	//! Check Dale's law; inhibitory; additive Gaussian noise
	EXCITATORY_DIRECT, 	//! Check Dale's law; excitatory; do not make any attempt to reinterpret input from this node
	INHIBITORY_DIRECT	//! Check Dale's law; inhibitory; do not make any attempt to reinterpret input
};
	//! Test for excitatoryness
	inline bool IsExcitatory(NodeType t){ return (t == EXCITATORY_DIRECT || t == EXCITATORY_GAUSSIAN) ? true : false; }

	//! Test for inhibitoryness
	inline bool IsInhibitory(NodeType t){ return (t == INHIBITORY_DIRECT || t == INHIBITORY_GAUSSIAN) ? true : false; }
} // end of namespace MPILib

#endif // include guard MPILIB_NODETYPE_HPP_
