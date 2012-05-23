// Copyright (c) 2005 - 2007 Marc de Kamps, Johannes Drever, Melanie Dietz
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

#include "Util.h"

using namespace LayerMappingLib;

/* JO code starts here */

vector<vector<int> > LayerMappingLib::combinations( int v, int l )
{
	if( l == 1 )
	{
		vector<vector<int> > r;
		for( int i = 0; i < v; i++ )
		{
			vector<int> x;
			x.push_back( i );
			r.push_back( x );
		}
		return r;
	}
	vector<vector<int> > temp = combinations( v, l - 1 );
	vector<vector<int> > r;


	for( int i = 0; i < v; i++ )
	{
		for( vector<vector<int> >::iterator vi = temp.begin();
			vi != temp.end();
			vi++ )
		{
			vector<int> x = *vi;
			x.push_back( i );
			r.push_back( x );
		}
	}
	return r;
}

// vector<vector<int> > LayerMappingLib::combinations_iterative( int v, int l )
// {
// 	int n = (int) pow( v, l );
// 	vector<vector<int> >r( n );
// 	
// 	int x = 0;
// 	for( vector<vector<int> >::iterator i = r.begin();
// 		i != r.end();
// 		i++, x++ )
// 	{
// 		i->resize( l );
// 		int p = 0;
// 		for( vector<int>::reverse_iterator j = i->rbegin();
// 			j != i->rend();
// 			j++, p++ )
// 		{
// 			*j = ( x / (int) pow( v, p ) ) % v;
// 		}
// 	}
// 	return r;
// }

/* JO code ends here */

