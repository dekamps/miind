// Copyright (c) 2005 - 2008 Marc de Kamps
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
#ifndef _CODE_LIBS_NETLIB_LOCALDEFINITIONS_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_LOCALDEFINITIONS_INCLUDE_GUARD


namespace NetLib
{
	const double F_DERIVATIVE_CUTOFF = 1e3; // absolute value beyond which the derivative of a squashing function is set to zero

	const string STR_ABSTRACTNODELINKCOLLECTION_TAG
	(
		"AbstractNodeLinkCollection - seeing this tag is probably due to error"
	);

	const string STR_LAYEREDIMPLEMENTATION_TAG
	(
		"<LayeredImplementation>"
	);

	const string STR_NOSQUASHINGFUNCTION_TAG
	(
		"<NoSquashingFunction>"
	);

	const string STR_NOSQUASHINGPARAMETER_TAG
	(
		"<NoSquashingParameter>"
	);

	const string STR_DOUBLE_REGISTER
	(
		"An attempt to register a registered squqshing function"
	);

	const string STR_SQUASH_EXCEPTION
	(
		"Unknown squashing function: is it registered ?"
	);

	const string STR_SIGMOID_HEADER
	(
		"<Sigmoid>"
	);

	const string STR_SIGMOID_HEADER_EXPECTED
	(
		"Expected Sigmoid header"
	);

	const string STR_SIGMOID_FOOTER_EXPECTED
	(
		"Expected Sigmoid footer"
	);

	const string STR_PATTERN_TAG             
	(
		"<Pattern>"
	);

	const string STR_FOOTER_EXCEPTION        
	(
		"Pattern footer expected"
	);

	const string STR_HEADER_EXCEPTION        
	(
		"Pattern header tag expected"
	);

	const string STR_INT_EXCEPTION           
	(
		"int exception"
	);

	const string STR_PATTERN_VALUE_EXCEPTION 
	(
		"Incorrect pattern value found"
	);

	const string STR_DLNLC
	(
		"<DefaultLayerNodeLinkCollection>"
	);
} // end of ImplementationLib

#endif // include guard
