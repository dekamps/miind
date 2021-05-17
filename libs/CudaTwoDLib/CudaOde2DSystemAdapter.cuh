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
#include <TwoDLib/Ode2DSystemGroup.hpp>
#include <map>

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
			TwoDLib::Ode2DSystemGroup& group, // The group must already be initialized. This will be checked and an exception will be thrown if it is suspected this has not happened
			unsigned int mesh_objects_start_index = 0
		);

		//! Standard Constructor
		CudaOde2DSystemAdapter
		(
			TwoDLib::Ode2DSystemGroup& group, // The group must already be initialized. This will be checked and an exception will be thrown if it is suspected this has not happened
		    double network_time_step,
			unsigned int mesh_objects_start_index = 0
		);


		~CudaOde2DSystemAdapter();


        void Evolve();

		void Evolve(std::vector<inttype>& meshes);

		void EvolveWithoutMeshUpdate();

		void EvolveOnDevice();

		void EvolveOnDevice(std::vector<inttype>& meshes);

		void EvolveWithoutTransfer();

		void EvolveWithoutTransfer(std::vector<inttype>& meshes);

        void Dump(const std::vector<std::ostream*>&, int mode = 0);

        void RemapReversal();

		void RemapReversalFiniteObjects();

		void RedistributeProbability();

		void RedistributeProbability(std::vector<inttype>& meshes);

		void RedistributeFiniteObjects(double timestep, curandState* rand_state);

		void RedistributeFiniteObjects(std::vector<inttype>& meshes, double timestep, curandState* rand_state);

		void RedistributeGridFiniteObjects(curandState* rand_state);

		void RedistributeGridFiniteObjects(std::vector<inttype>& meshes, curandState* rand_state);

		void MapFinish();

		void MapFinish(std::vector<inttype>& meshes);

		void updateGroupMass();

		void updateFiniteObjects();

		void TransferFiniteObjects();

		void updateRefractory();

		void UpdateMapData();

		fptype sumRefractory();

		inttype* getSpikes() { return _spikes; }

		friend class CSRAdapter;
								
		MPILib::Potential getAvgV(unsigned int m);

        const std::vector<fptype>& F(unsigned int n_steps = 1) const;

		const unsigned int NumObjects() const { return _group.NumObjects(); }

		TwoDLib::Ode2DSystemGroup& getGroup() { return _group; }
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
		void EstimateGridThresholdsResetsRefractories(const std::vector<TwoDLib::Mesh>& vec_mesh,
			const std::vector<std::vector<TwoDLib::Redistribution> >& vec_vec_reset, const std::vector<MPILib::Time>& times);

		void FillFiniteVectors();

        void FillReversalMap(const std::vector<TwoDLib::Mesh>&, const std::vector<std::vector<TwoDLib::Redistribution> >&);
        void FillResetMap(const std::vector<TwoDLib::Mesh>&, const std::vector<std::vector<TwoDLib::Redistribution> >&);
				void FillRefractoryTimes(const std::vector<MPILib::Time>&);

        void DeleteMass();
		void DeleteFiniteVectors();
		void DeleteMapData();
		void DeleteReversalMap();
        void DeleteResetMap();
		void FillDerivative();
		void DeleteDerivative();
		void FillSpikesAndSpikeCounts();

		TwoDLib::Ode2DSystemGroup& _group;
        inttype	 _n;
        inttype _mesh_size;
        fptype _time_step;
		fptype _network_time_step;

		std::vector<unsigned int> _nr_refractory_steps;
		std::vector<fptype> _refractory_prop;
		std::vector<fptype*> _refractory_mass;
		std::vector<std::vector<fptype>> _refractory_mass_local;
		std::vector<std::vector<fptype>>  _vec_alpha_ord;

		fptype*  _mass;
        std::vector<fptype> _hostmass;
        inttype* _map;
        std::vector<inttype> _hostmap;
		inttype* _unmap;
		std::vector<inttype> _hostunmap;
		inttype* _map_cumulative_value;
		std::vector<inttype> _host_map_cumulative_value;
		inttype* _map_strip_length_value;
		std::vector<inttype> _host_map_strip_length_value;
        std::vector<inttype> _offsets;

		// Finite size
		std::vector<inttype>  _vec_num_objects;
		std::vector<inttype>  _vec_num_object_offsets;
		// The index to _vec_num_objects and _vec_num_objects_offsets 
		//where the values refer to meshalgorithms instead of gridalgorithms.
		// Assumes that the above vectors were built grids then meshes which is guaranteed in VectorizedNetwork
		unsigned int		  _mesh_objects_start_index;
		inttype* _vec_objects_to_index;
		std::vector<inttype> _host_vec_objects_to_index;
		//std::vector<inttype*> _vec_cells_to_objects;
		//std::vector<std::vector<inttype>> _host_vec_cells_to_objects;
		fptype*  _vec_objects_refract_times;
		std::vector<fptype> _host_vec_objects_refract_times;
		inttype* _vec_objects_refract_index;
		std::vector<inttype> _host_vec_objects_refract_index;

		std::vector<fptype> _thresholds;
		std::vector<fptype> _resets;
		std::vector<fptype> _reset_ws;
		std::vector<fptype> _refractories;

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
		std::vector<inttype*> _res_to_ordered;
		std::vector<inttype*> _res_from_counts;
		std::vector<inttype*> _res_from_offsets;

		std::vector<fptype*>   _res_to_mass;
		std::vector<fptype*>   _res_sum;
		std::vector<inttype*>  _spikeCounts;
		inttype* _spikes;

		int _blockSize;
		int _numBlocks;

        // firing rates
        fptype* _fs;
        //technically this modifies the object, but the client shouldn't modify the frequencies
        mutable std::vector<fptype> _host_fs;

	};
}
#endif // include guard
