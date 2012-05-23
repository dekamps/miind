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

#include "ImageConverter.h"

using namespace std;

using namespace LayerMappingLib;
using namespace LayerMappingLib::gtkmm;

Glib::RefPtr<Gdk::Pixbuf> ImageConverter::pixbuf( FeatureMap<double>::iterator p, FeatureMap<double>::iterator p_end, int width, int height )
{
	double MAX_VALUE = 255.0;

	guchar* temp = new guchar[ width * height * 3 ];

	guchar* ti = temp;
	for( FeatureMap<double>::iterator p_i = p;
		p_i != p_end;
		p_i++, ti++ )
	{
		double x = *p_i * MAX_VALUE;
		*ti = (guchar) x; ti++;
		*ti = (guchar) x; ti++;
		*ti = (guchar) x;
	}

	Glib::RefPtr<Gdk::Pixbuf> r = Gdk::Pixbuf::create_from_data( temp,
		Gdk::COLORSPACE_RGB,
		false,
		8,
		width,
		height,
		width * 3,
		bind( boost::lambda::delete_array(), boost::lambda::_1 ) );
	return r;
}

void ImageConverter::featuremap( Gtk::Image& image, FeatureMap<double>& featuremap, int channel )
{
	Glib::RefPtr<Gdk::Pixbuf> pixbuf = image.get_pixbuf();

	assert( pixbuf->get_width() == featuremap.width() );
	assert( pixbuf->get_height()== featuremap.height() );
	assert( pixbuf->get_n_channels() == 3 );

	guint8* pix = pixbuf->get_pixels();

	FeatureMap<double>::iterator s_i = featuremap.begin();
	switch( channel )
	{
		guint8* pix_end;

		case 4: //combine channels
		pix_end = pix + pixbuf->get_height() * pixbuf->get_width() * 3;

		for( guint8* pix_i = pix;
			pix_i != pix_end;
			pix_i++, s_i++ )
		{
			guint8 avg = 0;

			avg += *pix_i; pix_i++;
			avg += *pix_i; pix_i++;
			avg += *pix_i;

			*s_i = *pix_i;
		}
		break;

		default:
		stringstream ss( stringstream::in | stringstream::out );

		ss << "channel " << channel << " does not exist." << endl;
		throw Exception( ss.str() );
	}
}
