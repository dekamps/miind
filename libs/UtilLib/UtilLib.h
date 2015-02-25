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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_UTILLIB_INCLUDE_GUARD
#define _CODE_LIBS_UTILLIB_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable:4786)
#endif // pragma


#include <vector>
#include <cmath>
#include <string>

#include "AttributeList.h"
#include "ColorTable.h"
#include "GeneralException.h"
#include "FailAlloc.h"
#include "IsNan.h"
#include "IsFinite.h"
#include "MToCm.h"
#include "NumericException.h"
#include "ParameterScan.h"
#include "Persistant.h"
#include "Point.h"
#include "PointerContainerIteratorCode.h"
#include "PositionInCm.h"
#include "LogStream.h"
#include "RGBValue.h"
#include "Shape.h"
#include "ConcreteStreamable.h"
#include "SequenceIteratorIterator.h"
#include "ToValarray.h"
#include "ToVector.h"


//! namespace Util

namespace UtilLib
{
	class  NoTimeErr 
	{
	};
	
	class RegisteryException 
	{
	};
	
	class Registery 
	{
		public:
		
			Registery( size_t, size_t );
			int Next();
			void ReserveRange( size_t, size_t );
			size_t NrValuesLeft() const;
			void GiveBack( size_t );
		
		private:
		
			size_t NrIndexFromValue( size_t ) const;
			size_t NrValueFromIndex( size_t ) const;
		
			std::vector<bool> _vec_reg;
			size_t _offset;
	};
}
#endif // include guard 
