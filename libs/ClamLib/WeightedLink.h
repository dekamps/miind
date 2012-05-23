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

#ifndef _CODE_LIBS_CLAMLIB_WEIGHTEDLINK_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_WEIGHTEDLINK_INCLUDE_GUARD

#include <TNamed.h>
#include "Id.h"

//namespace ClamLib {

//! This is a ClamLib version of WeightedLink and a duplication of
//! DynamicLib::WeightedLink. This version is copied to enable ROOT serialization
//! but should be removed in the long run (TODO)

	template <class Weight>
	class WeightedLink : public TNamed
	{
	public:
		ClassDef(WeightedLink,1);

		ClamLib::Id _id_from;
		ClamLib::Id _id_to;
		double _weight;

		WeightedLink():_id_from(ClamLib::Id(0)),_id_to(ClamLib::Id(0)),_weight(0.0){}

		WeightedLink
		(
			ClamLib::Id		id_from,
			ClamLib::Id		id_to,
			double 			weight
		):
		_id_from(id_from),
		_id_to(id_to),
		_weight(weight)
		{
		}
	};

	typedef WeightedLink<double>   D_WeightedLink;

//}

#endif // include guard
