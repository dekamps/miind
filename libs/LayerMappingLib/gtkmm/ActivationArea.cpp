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

#include "ActivationArea.h"

using namespace LayerMappingLib::gtkmm;

ActivationArea::ActivationArea() :
		_image ( NULL ),
		_featuremap()
{
}

ActivationArea::ActivationArea ( const ActivationArea& a ) : _featuremap ( a._featuremap )
{
	DrawingArea::set_size_request( _featuremap.width(), _featuremap.height() );
}

ActivationArea::~ActivationArea()
{}

void ActivationArea::operator=( const ActivationArea& s )
{
	_featuremap = s._featuremap;
	DrawingArea::set_size_request( _featuremap.width(), _featuremap.height() );
}

void ActivationArea::setFeatureMap( FeatureMap<double> l )
{
	_featuremap = l;
	DrawingArea::set_size_request( _featuremap.width(), _featuremap.height() );
}

void ActivationArea::show()
{
	Gtk::DrawingArea::show();
}

bool ActivationArea::on_expose_event( GdkEventExpose* event )
{
	_image = ImageConverter::pixbuf( _featuremap.begin(),
					_featuremap.end(),
					_featuremap.width(),
					_featuremap.height() );

	_image->render_to_drawable ( get_window(),
	                             get_style()->get_black_gc(),
	                             0, 0,
	                             0, 0,
	                             _featuremap.width(), _featuremap.height(),
	                             Gdk::RGB_DITHER_NONE, 0, 0 );

	return true;
}

