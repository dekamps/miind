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
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSEIMPLEMENTATIONALLOCATOR_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSEIMPLEMENTATIONALLOCATOR_INCLUDE_GUARD

#include "SparseNode.h"
#include "ReversibleSparseNode.h"


namespace DynamicLib {
	template <class Weight > class DynamicNode;
}

namespace SparseImplementationLib {


#include <limits>
#include <memory>
using std::allocator;
using std::numeric_limits;


using SparseImplementationLib::SparseNode;


	//! Serves to make collections of derived classes of AbstractSparseNodes.
	//!
	//! A SparseImplementation is a collection of nodes. These nodes derive from AbstractSparseNode, and internally these nodes
	//! maintain pointer-weight pairs to other nodes. The type of the pointer is AbstractSparseNode*, so pointers to the base
	//! class.
	template <typename T>
	class SparseImplementationAllocator {
	public : 
		//    typedefs
		typedef T value_type;
		typedef value_type* pointer;
		typedef const value_type* const_pointer;
		typedef value_type& reference;
		typedef const value_type& const_reference;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;

	public : 
		//    convert an allocator<T> to allocator<U>
		template<typename U>
	struct rebind {
		typedef SparseImplementationAllocator<U> other;
	};

	public : 
		inline explicit SparseImplementationAllocator() {}
		inline ~SparseImplementationAllocator() {}
		inline SparseImplementationAllocator(SparseImplementationAllocator const&) {}
    
		template<typename U>
		inline SparseImplementationAllocator(SparseImplementationAllocator<U> const&) {}

		//    address
		inline pointer address(reference r) { return &r; }
		inline const_pointer address(const_reference r) { return &r; }

		//    memory allocation
		void deallocate(pointer p, size_type cnt);

		pointer allocate
		(
			size_type cnt, 		
			typename std::allocator<void>::const_pointer = 0
		);

		//    size
		inline size_type max_size() const { 
			return std::numeric_limits<size_type>::max() / sizeof(T);
		}

		//    construction/destruction
//		inline void construct(pointer p, const T& t) { new(p) T(t); p->ApplyOffset(&t); }
#ifdef WIN32
		inline void construct(pointer p, const std::_Container_proxy& t){ new(p) T(t); }
#endif
		template <class W, class A> inline void construct(pointer p, const SparseNode<W,A>& t){new(p) T(t); p->ApplyOffset(&t); }
		template <class W, class A> inline void construct(pointer p, const ReversibleSparseNode<W,A>& t){new(p) T(t); p->ApplyOffset(&t); }
		template <class Weight> inline void construct(pointer p, const DynamicLib::DynamicNode<Weight>& t){new(p) T(t); p->ApplyOffset(&t); }

		inline void destroy(pointer p) { p->~T(); }

		inline bool operator==(SparseImplementationAllocator const&) { return true; }
		inline bool operator!=(SparseImplementationAllocator const& a) { return !operator==(a); }

	private:

		allocator<T>	_allocator;
	
	};    //    end of class SparseImplementationAllocator  

	template <class T>
	inline typename SparseImplementationAllocator<T>::pointer SparseImplementationAllocator<T>::allocate
	(
		size_type cnt, 		
		typename std::allocator<void>::const_pointer
	) 
	{
		return _allocator.allocate(cnt);
	}


	template <class T>
	void SparseImplementationAllocator<T>::deallocate
	(
		pointer p, 
		size_type cnt
	) 
	{
		_allocator.deallocate(p,cnt);
	}

}

#endif // include guard
