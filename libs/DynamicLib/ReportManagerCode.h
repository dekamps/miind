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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_LIBS_DYNAMICLIB_REPORTMANAGERCODE_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_REPORTMANAGERCODE_INCLUDE_GUARD
/*
#include "ReportManager.h"

namespace DynamicLib
{

	template <class WeightValue>
	ReportManager<WeightValue>::ReportManager
	(
		const DynamicNetworkImplementation<WeightValue>& implementation
	):
	_p_implementation(const_cast<DynamicNetworkImplementation<WeightValue>*>(&implementation))
	{
	}

	template <class WeightValue>
	ReportManager<WeightValue>::~ReportManager()
	{
	}



	template <class WeightValue>
	bool ReportManager<WeightValue>::OpenReport() const
	{
		typedef typename vector< DynamicNode<WeightValue> >::iterator node_iterator;

		node_iterator iter_begin = _p_implementation->begin();
		node_iterator iter_end   = _p_implementation->end();

		for_each
		(
			iter_begin,
			iter_end,
			mem_fun_ref(&DynamicNode<WeightValue>::ReportOpen)
		);	
	}

	template <class WeightValue>
	bool ReportManager<WeightValue>::CloseReport() const
	{
		typedef typename vector< DynamicNode<WeightValue> >::iterator node_iterator;

		node_iterator iter_begin = _p_implementation->begin();
		node_iterator iter_end   = _p_implementation->end();

		for_each
		(
			iter_begin,
			iter_end,
			mem_fun_ref(&DynamicNode<WeightValue>::ReportClose)
		);

		return true;
	}

} // end of DynamicLib
*/
#endif // include guard
