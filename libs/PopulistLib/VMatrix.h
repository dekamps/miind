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
#ifndef _CODE_LIBS_POPULULISTLIB_VMATRIX_INCLUDE_GUARD 
#define _CODE_LIBS_POPULULISTLIB_VMATRIX_INCLUDE_GUARD 

#include <string>
#include "../NumtoolsLib/NumtoolsLib.h"
#include "BasicDefinitions.h"

using NumtoolsLib::C_Matrix;
using NumtoolsLib::D_Matrix;
using std::string;


namespace PopulistLib {

	enum CalculationMode {FULL_CALCULATION, LOOK_UP};

	//! This is a class which can compute the complex incomplete gamma function and from these the
	//! parameters V_kj, the matrix elements for the circulant solution. For practical use this
	//! class is DECREPATED and has been replaced by VArray.
	template <CalculationMode mode>
	class VMatrix
	{
	public:

		VMatrix();


		bool GenerateVLookup
		(
			const string&,
			Number,
			Number,
			Time,
			Number
		);

		double V
		(
			Number,
			Index,
			Index,
			Time
		);

		bool FillGammaZ
		(
			Number,
			Number,
			Time
		);

		bool CalculateV_kj
		(
			Number,
			Number,
			Time
		);


		complex<double> Gamma
		(
			Index,
			Index
		) const ;

	private:

		Number NumberOfMatrixElements() const;

		double VFullCalculation
			(
					Number,
					Index,
					Index,
					Time
			);

		double VLookUp
			(
					Number,
					Index,
					Index,
					Time
			);


		template <class Matrix> 
			bool FillMatrixWithGarbage
				(
					Matrix&
				);


		Number				_number_current_circulant;
		Time				_tau_current;
		C_Matrix			_matrix_gamma_z;
		D_Matrix			_matrix_V_kj;
		valarray<double>	_array_faculty;

	};
}

#endif // include guard
