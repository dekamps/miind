// Copyright (c) 2005 - 2011 Marc de Kamps
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
#ifndef _CODE_LIBS_POPULISTLIB_VMATRIXCODE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_VMATRIXCODE_INCLUDE_GUARD

#include <fstream>
#include <gsl/gsl_math.h>
#include "VMatrix.h"
#include "LocalDefinitions.h"
#include "PopulistException.h"

namespace PopulistLib {

	template <CalculationMode mode>
	Number VMatrix<mode>::NumberOfMatrixElements() const
	{
		return (mode == FULL_CALCULATION) ? MAXIMUM_NUMBER_GAMMAZ_VALUES + 1 : 0;
	}

	template <CalculationMode mode>
	inline VMatrix<mode>::VMatrix():
	_number_current_circulant
	(
		std::numeric_limits<Number>::max()
	),
	_tau_current
	(
		std::numeric_limits<double>::max()
	),
	_matrix_gamma_z
	(
		NumberOfMatrixElements(),
		NumberOfMatrixElements()
	),
	_matrix_V_kj
	(
		NumberOfMatrixElements(),
		NumberOfMatrixElements()
	),
	_array_faculty
	(
		NumberOfMatrixElements()
	)
	{

		if (_array_faculty.size() > 0 )
		{
			_array_faculty[1] = _array_faculty[0] = 1.0;
			for
			(
				int index = 2;
				index < static_cast<int>(_array_faculty.size());
				index++
			)
				_array_faculty[index] = index*_array_faculty[index-1];
		}
		// else
		//   nothing
	}

	template <CalculationMode mode>
	double VMatrix<mode>::V 
			(
				Number number_of_circulant_bins,
				Index  index_circulant,
				Index  index_non_circulant,
				Time   time
			) 
	{
		if ( mode == FULL_CALCULATION )
			return VFullCalculation
					(
						number_of_circulant_bins,
						index_circulant,
						index_non_circulant,
						time
					);
		else
			return  VLookUp
					(
						number_of_circulant_bins,
						index_circulant,
						index_non_circulant,
						time
					);

		return 0;
	}

	template <CalculationMode mode>
	double VMatrix<mode>::VFullCalculation
							(
								Number number_of_circulant_bins,
								Index  index_circulant,
								Index  index_non_circulant,
								Time   tau
							) 
	{

	// remember that GammaZ must be calculated with high precision

/*	if ( 
			number_of_non_circulant_areas >= MAXIMUM_NUMBER_GAMMAZ_VALUES ||
			number_of_circulant_bins      >= MAXIMUM_NUMBER_GAMMAZ_VALUES
		)
		return false;
*/
		if 
		(
			number_of_circulant_bins != _number_current_circulant ||
			tau                      != _tau_current
		)
		{
			FillGammaZ
			(
				MAXIMUM_NUMBER_GAMMAZ_VALUES, 
				number_of_circulant_bins,
				tau
			);

			CalculateV_kj
			(
				MAXIMUM_NUMBER_GAMMAZ_VALUES,
				number_of_circulant_bins,
				tau
			);

			_number_current_circulant = number_of_circulant_bins;
			_tau_current              = tau;
		}

		return _matrix_V_kj(index_circulant, index_non_circulant);
	}

	template <CalculationMode mode>
	double VMatrix<mode>::VLookUp
							(
								Number number_of_circulant_bins,
								Index  index_circulant,
								Index  index_non_circulant,
								Time   tau
							) 
	{
		throw PopulistException(STR_LOOKUP_DISABELED);
		return 0;
	}

	template <CalculationMode mode>
	bool VMatrix<mode>::FillGammaZ
	(
		Number number_of_non_circulant_areas,
		Number number_of_circulant_bins,
		Time   tau
	)
	{
		assert (FillMatrixWithGarbage(_matrix_gamma_z));

		if ( number_of_non_circulant_areas > static_cast<Number>(MAXIMUM_NUMBER_GAMMAZ_VALUES) )
		return false;

		for
		(
			int power = 0;
			power < static_cast<int>(number_of_circulant_bins);
			power++
		)
		{
			complex<double> omega_kt = exp
										(
											complex<double>(power*2.0*M_PI/number_of_circulant_bins,0.0)*complex<double>(0.0,1.0)
										)*complex<double>(tau,0);
			complex<double> ksi = pow(omega_kt,number_of_non_circulant_areas)*exp(-omega_kt)/complex<double>(number_of_non_circulant_areas,0);


			complex<double> n1 =    complex<double>(number_of_non_circulant_areas + 2,0);
			complex<double> n2 = n1*complex<double>(number_of_non_circulant_areas + 3,0);
			complex<double> n3 = n2*complex<double>(number_of_non_circulant_areas + 4,0);
			complex<double> n4 = n3*complex<double>(number_of_non_circulant_areas + 5,0);
			complex<double> n5 = n4*complex<double>(number_of_non_circulant_areas + 6,0);

			complex<double> o1 = omega_kt;
			complex<double> o2 = omega_kt*o1;
			complex<double> o3 = omega_kt*o2;
			complex<double> o4 = omega_kt*o3;
			complex<double> o5 = omega_kt*o4;

			complex<double> gamma_seed = ksi*( complex<double>(1,0) + o1/n1 + o2/n2 + o3/n3 + o4/n4 + o5/n5 );


			_matrix_gamma_z(number_of_non_circulant_areas,power) = gamma_seed/_array_faculty[number_of_non_circulant_areas-1];
			for 
			(
				int j = number_of_non_circulant_areas - 1;
				j > 0;
				j--
			)
				{
					gamma_seed = (gamma_seed + pow(omega_kt,j)*exp(-omega_kt) )/complex<double>(j,0);
					_matrix_gamma_z(j, power) = gamma_seed/_array_faculty[j-1];
				}
		}
		return true;
	}

	template <CalculationMode mode>
	complex<double> VMatrix<mode>::Gamma
								(
									Index j,
									Index power
								) const
	{
		assert( j > 0);
		assert(     j <= MAXIMUM_NUMBER_GAMMAZ_VALUES);
		assert( power <= MAXIMUM_NUMBER_GAMMAZ_VALUES);

		return _matrix_gamma_z(j,power);
	}

	template <CalculationMode mode>
	template <class Matrix> 
	inline bool VMatrix<mode>::FillMatrixWithGarbage(Matrix& matrix)
	{
		assert
		( 
			matrix.NrXdim() == MAXIMUM_NUMBER_GAMMAZ_VALUES + 1 && 
			matrix.NrYdim() == MAXIMUM_NUMBER_GAMMAZ_VALUES + 1
		);

		for (int ix = 0; ix < MAXIMUM_NUMBER_GAMMAZ_VALUES; ix++ )
			for (int iy = 0; iy < MAXIMUM_NUMBER_GAMMAZ_VALUES; iy++)
				matrix(ix, iy) = 9.99999999e99;

		return true;
	}

	template <CalculationMode mode>
	bool VMatrix<mode>::CalculateV_kj
						(
							Number number_of_non_circulant_areas,
							Number number_of_circulant_bins,
							Time   tau
						)
	{
		assert( FillMatrixWithGarbage(_matrix_V_kj) );

		if 
		( 
		 number_of_non_circulant_areas > static_cast<Number>(MAXIMUM_NUMBER_GAMMAZ_VALUES) ||
		 number_of_circulant_bins      > static_cast<Number>(MAXIMUM_NUMBER_GAMMAZ_VALUES) 
		)
			return false;

		int n_half = (number_of_circulant_bins - 1)/2;
		for 
		( 
			int index_circulant = 0;
			index_circulant < static_cast<int>(number_of_circulant_bins);
			index_circulant++
		)
		{
			for 
			(
				int index_non_circulant = 0;
				index_non_circulant < static_cast<int>(number_of_non_circulant_areas);
				index_non_circulant++
			)
				{
					_matrix_V_kj
						(
							index_circulant,
							index_non_circulant
						) = Gamma(index_non_circulant + 1, 0).real()/number_of_circulant_bins;

					for 
					(
						int index_pair = 1;
						index_pair <= n_half;
						index_pair++
					)
						{
							double argument = 2*M_PI*index_pair*(index_circulant + index_non_circulant + 1)/number_of_circulant_bins -
								              sin(2*M_PI*index_pair/number_of_circulant_bins)*tau;

							double real_gamma = Gamma
												(
													index_non_circulant + 1, 
													index_pair
												).real();

	
							double imag_gamma = Gamma
												(
													index_non_circulant + 1, 
													index_pair
												).imag();

							double factor = 2*exp( (cos(index_pair*2.0*M_PI/number_of_circulant_bins) - 1)*tau)/number_of_circulant_bins;

							_matrix_V_kj
								( 
									index_circulant,
									index_non_circulant
								) += factor*( cos(argument)*real_gamma + sin(argument)*imag_gamma );
		
								
						}
					if ( number_of_circulant_bins%2 ==  0)
						{
							double odd_or_even = ( (index_circulant + index_non_circulant)%2 == 0 ) ? -1.0 : 1.0;
						
							_matrix_V_kj 
								(
									index_circulant,
									index_non_circulant
								) += odd_or_even*exp(-2.0*tau)*Gamma
																(
																	index_non_circulant + 1, 
																	number_of_circulant_bins/2
																).real()/number_of_circulant_bins;
						}
				}
		}

		return true;
	}

	template <CalculationMode mode>
	bool VMatrix<mode>::GenerateVLookup
						(
							const string& name_directory,
							Number        number_maximum_circulant_bins,
							Number        number_maximum_non_circulant_areas,
							Time          tau_max,
							Number        number_time_steps
						)
	{


	Time time_step = tau_max/static_cast<double>(number_time_steps);


	string str_h = name_directory + STR_VLOOKUP_H;
	ofstream stream_h(str_h.c_str());

	stream_h << "#ifndef _CODE_LIBS_POPULISTLIB_VLOOKUP_INCLUDE_GUARD\n";
	stream_h << "#define _CODE_LIBS_POPULISTLIB_VLOOKUP_INCLUDE_GUARD\n\n";

	stream_h << "#include \"../Util/Util.h\" "     << "\n";
	stream_h << "#include \"TestDefinitions.h\" "  << "\n\n";
	stream_h << "using Util::Number;\nusing Util::Index;\n\n";

	stream_h << "namespace PopulistLib {\n\n";

	stream_h << "\tconst Number NUMBER_MAXIMUM_CIRCULANT     = " << number_maximum_circulant_bins      << ";\n";
	stream_h << "\tconst Number NUMBER_MAXIMUM_NON_CIRCULANT = " << number_maximum_non_circulant_areas << ";\n";
	stream_h << "\tconst Time   TIME_MAX  =\t"                   << tau_max                            << ";\n";
	stream_h << "\tconst Time   TIME_STEP =\t"                   << time_step                          << ";\n\n";

	stream_h << "\tinline int ConvertSumIndexToIndexCirculant(int index_sum){\n";
	stream_h << "\t\treturn ( index_sum < NUMBER_MAXIMUM_CIRCULANT ) ? index_sum : NUMBER_MAXIMUM_CIRCULANT - 1;\n";
	stream_h << "\t}\n\n";

	stream_h << "\tinline int ConvertSumIndexToIndexNonCirculant(int index_sum){\n";
	stream_h << "\t\treturn ( index_sum < NUMBER_MAXIMUM_CIRCULANT ) ? 0 : index_sum - (NUMBER_MAXIMUM_CIRCULANT - 1);\n";
	stream_h << "\t}\n\n";


	stream_h << "\tinline int ToArrayIndex(Number number_circulant, Index index_circulant, Index index_non_circulant){\n";
	stream_h << "\t\treturn index_circulant + index_non_circulant  + (number_circulant - 1)*NUMBER_MAXIMUM_NON_CIRCULANT + (number_circulant - 1)*(number_circulant )/2" << ";\n\t}\n\n";

	stream_h << "\tdouble VLookup\n(\n\t\t Number number_circulant, \n\t\t Index i_circulant,\n\t\t Index i_non_circulant,\n\t\t Time time\n\t);\n";
	stream_h << "\tbool VMatrixAlloc();\n";

	stream_h << "\n}\n\n";


	stream_h << "#endif // include guard\n";

	string str_cpp = name_directory + STR_VLOOKUP_CPP;
	ofstream stream_cpp(str_cpp.c_str());

	if (! stream_cpp )
		return false;

	stream_cpp.precision(12);

	stream_cpp << "#include \"" << STR_VLOOKUP_H << "\"" << endl;
	stream_cpp << "#include <gsl/gsl_spline.h>"  << endl;

	stream_cpp << "using namespace std;\nusing namespace PopulistLib;\n\n";
	stream_cpp << "namespace {" << endl;



	stream_cpp << "\tconst double t_v[] = { ";

	for 
	(
		Time time_v =  0;
		time_v      <  number_time_steps*time_step;
		time_v      += time_step
	)
	{
		string str_marker = (time_v == 0) ? " " : ", ";
		stream_cpp << str_marker << time_v;
	}
	stream_cpp << " };" << endl << endl;



	for
	(
		int number_circulant = 1;
		number_circulant     < static_cast<int>(number_maximum_circulant_bins);
		number_circulant++
	)
	{

		cout << number_circulant << endl;

		int max_sum_circulant_non_circulant = number_circulant + number_maximum_non_circulant_areas;
		for
		( 
			int index_sum_k_j = 0;
			index_sum_k_j     < max_sum_circulant_non_circulant;
			index_sum_k_j++
		)
		{

			ostringstream array_name, array_V;

			array_name << "V_" << number_circulant <<  "_" << number_maximum_non_circulant_areas << "_" << index_sum_k_j;				
			stream_cpp << "\tconst double " << array_name.str() << "[] = {"; 

			cout << array_name.str() <<  " " << index_sum_k_j  + (number_circulant -1)*number_maximum_non_circulant_areas + (number_circulant - 1)*(number_circulant )/2<< "\n";



			for 
			(
				Time time = 0;
				time <  number_time_steps*time_step;
				time += time_step
			)
			{

				string str_marker = (time == 0 ) ? " " : ",";

				int index_circulant     = (index_sum_k_j < static_cast<int>(number_maximum_non_circulant_areas) ) ? 
					                       0 : 
									       index_sum_k_j - (number_maximum_non_circulant_areas - 1);


			    int index_non_circulant = (index_sum_k_j < static_cast<int>(number_maximum_non_circulant_areas) ) ? 
										   index_sum_k_j : 
					                       number_maximum_non_circulant_areas - 1;

				stream_cpp << str_marker << this->V
												(
													number_maximum_circulant_bins,
													index_circulant,
													index_non_circulant, 
													time
												);
			}

			stream_cpp << "};\n";
		}				
	}

	for
	( 
		int n_c = 1;
		n_c < static_cast<int>(number_maximum_circulant_bins);
		n_c++
	)
	{
		for 
		(
			int i_sum = 0;
			i_sum < static_cast<int>(number_maximum_non_circulant_areas) + n_c;
			i_sum++
		)
		{
			ostringstream array_name;
			array_name << "SPLINE_" << n_c <<  "_" << number_maximum_non_circulant_areas << "_" << i_sum;	
			stream_cpp <<"\tgsl_spline* " << array_name.str()<< ";\n";
		}
	}

	stream_cpp << "\n\tvector<gsl_spline*> array_pointer(0);\n";
/*	for 
	( 
		int number_c = 1;
		number_c < static_cast<int>(number_maximum_circulant_bins);
		number_c++
	)
	{

		for
		(
			int index_sum = 0;
			index_sum     < static_cast<int>(number_maximum_non_circulant_areas) +number_c;
			index_sum++
		)
		{
			ostringstream array_name;
			array_name << "SPLINE_" << number_c <<  "_" << number_maximum_non_circulant_areas << "_" << index_sum;				
			string str_marker = (number_c == 1 && index_sum == 0) ? " " : ", "; 
			stream_cpp << str_marker << array_name.str();
		}
		
	}
			
	stream_cpp << "};\n\n";
*/
	stream_cpp << "}\n\n";

	stream_cpp << "gsl_interp_accel* P_ACCELERATOR;\n\n";

	stream_cpp << "\tbool PopulistLib::VMatrixAlloc() {\n";
	stream_cpp << "\t\tP_ACCELERATOR = gsl_interp_accel_alloc ();\n";
	stream_cpp << "\t\tNumber number_of_time_steps = static_cast<Number>(TIME_MAX/TIME_STEP);\n";
	stream_cpp << "\t\tarray_pointer.clear();\n";
	for 
	( 
		int number_circu = 1;
		number_circu < static_cast<int>(number_maximum_circulant_bins);
		number_circu++
	)
	{

		for
		(
			int index_SUM = 0;
			index_SUM     < static_cast<int>(number_maximum_non_circulant_areas) + number_circu;
			index_SUM++
		)
		{
			ostringstream array_name, array_V;
			array_name <<  number_circu <<  "_" << number_maximum_non_circulant_areas << "_" << index_SUM;
			array_V    << "V_" << number_circu << "_" << number_maximum_non_circulant_areas << "_" << index_SUM;
			stream_cpp << "\t\t" << "SPLINE_" << array_name.str() << " = gsl_spline_alloc (gsl_interp_akima, number_of_time_steps);\n";
			stream_cpp << "\t\tgsl_spline_init (SPLINE_" << array_name.str() << ", t_v, " << array_V.str() << ", number_of_time_steps);\n";
			stream_cpp << "\t\tarray_pointer.push_back(" << "SPLINE_" << array_name.str() << ");\n";

		}
		
	}
	stream_cpp << "\t\treturn true;\n";

	stream_cpp << "\t}\n\n";


	stream_cpp << "double PopulistLib::VLookup\n(\n\t Number number_circulant, \n\t Index i_circulant,\n\t Index i_non_circulant,\n\t Time time\n)\n{\n";

	stream_cpp << "\tassert( i_non_circulant  <  NUMBER_MAXIMUM_NON_CIRCULANT);\n";
	stream_cpp << "\tassert( i_circulant      <  number_circulant);\n";
	stream_cpp << "\tassert( number_circulant <= NUMBER_MAXIMUM_NON_CIRCULANT);\n";
	stream_cpp << "\tassert( time             <= TIME_MAX);\n";

	stream_cpp << "\n";
	stream_cpp << "\tconst gsl_spline* p_interpolation_array = array_pointer[ToArrayIndex(number_circulant,i_circulant,i_non_circulant)];\n";


			
	stream_cpp << "\tdouble v_return = gsl_spline_eval (p_interpolation_array, time, P_ACCELERATOR);\n";


	stream_cpp << "\treturn v_return;\n";
	stream_cpp << "}\n\n";



		return true;
	}
}

#endif // include guard

