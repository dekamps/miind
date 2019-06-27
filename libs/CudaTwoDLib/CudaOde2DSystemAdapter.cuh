// Copyright (c) 2005 - 2015 Marc de Kamps
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

#ifndef _CODE_CUDA2DLIB_CudaOde2DSystemAdapter_INCLUDE_GUARD
#define _CODE_CUDA2DLIB_CudaOde2DSystemAdapter_INCLUDE_GUARD

#include <cassert>
#include "../TwoDLib/TwoDLib.hpp"


namespace CudaTwoDLib {

	  /**
	   * \brief  Responsible for maintaining the mirror of an Ode2DSystemGroup on a GPGPU device
	   *
	   * Maintains and when necessary synchronizes the mass array and the current mapping of the Ode2DGroup on the device
	   */

        //! floating point type for cuda system
	typedef float fptype;
	typedef unsigned int inttype;

	class CudaOde2DSystemAdapter {
	public:

		CudaOde2DSystemAdapter
		(
	           TwoDLib::Ode2DSystemGroup& group // The group must already be initialized. This will be checked and an exception will be thrown if it is suspected this has not happened
		);

		//! Standard Constructor
		CudaOde2DSystemAdapter
		(
	           TwoDLib::Ode2DSystemGroup& group, // The group must already be initialized. This will be checked and an exception will be thrown if it is suspected this has not happened
				 		MPILib::Time network_time_step

		);


		~CudaOde2DSystemAdapter();


                void Evolve();

								void Evolve(std::vector<inttype>& meshes);

								void EvolveWithoutMeshUpdate();

                void Dump(const std::vector<std::ostream*>&, int mode = 0);

                void RemapReversal();

								void RedistributeProbability();

								void RedistributeProbability(std::vector<inttype>& meshes);

								void MapFinish();

								void MapFinish(std::vector<inttype>& meshes);

								void updateGroupMass();

								void updateRefractory();

								fptype sumRefractory();

								friend class CSRAdapter;

                const std::vector<fptype>& F(unsigned int n_steps = 1) const;

								const TwoDLib::Ode2DSystemGroup& getGroup() { return _group; }
	private:

               CudaOde2DSystemAdapter(const CudaOde2DSystemAdapter&);
               CudaOde2DSystemAdapter& operator=(const CudaOde2DSystemAdapter&);

               struct MapElement {
                  inttype _to;
                  inttype _from;
                  fptype  _alpha;
               };

	void Validate() const;

        void FillMass();
        void FillMapData();
        void TransferMapData();

        void FillReversalMap(const std::vector<TwoDLib::Mesh>&, const std::vector<std::vector<TwoDLib::Redistribution> >&);
        void FillResetMap(const std::vector<TwoDLib::Mesh>&, const std::vector<std::vector<TwoDLib::Redistribution> >&);
				void FillRefractoryTimes(const std::vector<MPILib::Time>&);

        void DeleteMass();
				void DeleteMapData();
				void DeleteReversalMap();
        void DeleteResetMap();
				void FillDerivative();
				void DeleteDerivative();

				TwoDLib::Ode2DSystemGroup& _group;
        inttype	 _n;
        inttype _mesh_size;
        fptype _time_step;
				fptype _network_time_step;

				std::vector<unsigned int> _nr_refractory_steps;
				std::vector<fptype> _refractory_prop;
				std::vector<fptype*> _refractory_mass;
				std::vector<std::vector<fptype>> _refractory_mass_local;

				fptype*  _mass;
        std::vector<fptype> _hostmass;
        inttype* _map;
        std::vector<inttype> _hostmap;
        std::vector<inttype> _offsets;

        // reversal mapping
        inttype  _n_rev;
        inttype* _rev_to;
        inttype* _rev_from;
        fptype*  _rev_alpha;

        // reset mapping
        std::vector<inttype> _nr_resets;
				std::vector<inttype>  _nr_minimal_resets;
				std::vector<inttype*> _res_to_minimal;
				std::vector<fptype*>  _res_alpha_ordered;
				std::vector<inttype*> _res_from_ordered;
				std::vector<inttype*> _res_from_counts;
				std::vector<inttype*> _res_from_offsets;

				std::vector<fptype*>   _res_to_mass;
				std::vector<fptype*>   _res_sum;

				int _blockSize;
				int _numBlocks;

        // firing rates
        fptype* _fs;
        //technically this modifies the object, but the client shouldn't modify the frequencies
        mutable std::vector<fptype> _host_fs;

	};
}
#endif // include guard
