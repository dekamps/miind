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

#ifndef LAYERMAPPINGGTKMMLIB_NETWORKWIDGET_H
#define LAYERMAPPINGGTKMMLIB_NETWORKWIDGET_H

#include <gtkmm.h>
#include <gtkmm/window.h>
#include <gtkmm/box.h>

#include "LayerWidget.h"

#include "../FeatureMapNetworkCode.h"

using namespace LayerMappingLib;

namespace LayerMappingLib
{
	namespace gtkmm
	{
		class NetworkWidget : public Gtk::Frame
		{
			public:
			NetworkWidget( const string& title );
			virtual ~NetworkWidget();
			
			void add( FeatureMapNetwork<double>::iterator l_begin, FeatureMapNetwork<double>::iterator l_end, const string& title );
		
			void set_title( const string& s );
	
			protected:
			Gtk::ScrolledWindow _scrolled_window;
			Gtk::HBox _hbox;
		};
	}
}
#endif //LAYERMAPPINGGTKMMLIB_NETWORKWIDGET_H

