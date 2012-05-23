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

#ifndef LAYERMAPPINGGTKMMLIB_LAYERMAPPINGAPPLET_H
#define LAYERMAPPINGGTKMMLIB_LAYERMAPPINGAPPLET_H

#include <iostream>

#include <gtkmm.h>
#include <gtkmm/button.h>
#include <gtkmm/window.h>
#include <gtkmm/box.h>
#include <gtkmm/menubar.h>
#include <gtkmm/menu.h>
#include <gtkmm/stock.h>
#include <gtkmm/filechooserdialog.h>

#include "EnsembleWidget.h"
#include "../NetworkEnsembleCode.h"
#include "../algorithm.h"

using namespace LayerMappingLib;
using namespace std;

namespace LayerMappingLib
{
	namespace gtkmm
	{
		class LayerMappingApplet : public Gtk::Window
		{
			public:
			LayerMappingApplet();
	
	// 		Gtk::Image* loadImage( string s );
			void setNetworkEnsemble( NetworkEnsemble<double>& network_ensemble );
			void set_feedback_vector( vector<double>& v );

			virtual void show();
		
			virtual void on_button_evolve_clicked();
			virtual void on_button_set_feedback_activation_clicked();
	
			protected:
			virtual void on_filemenu_quit();
			virtual void on_filemenu_load();
	
			void _redraw();

			Gtk::VBox _vbox;
	
			Gtk::MenuBar _menubar;
			Gtk::Menu _filemenu;

			Gtk::Table _table;
			Gtk::VBox _box_button;
			Gtk::Button _button_evolve;
			Gtk::Button _button_set_feedback_activation;

	// 		Gtk::Image _image;
	
			EnsembleWidget _ensemble_widget;
	
			NetworkEnsemble<double> _network_ensemble;

			vector<double> _feedback_vector;
		};
	}
}
#endif //LAYERMAPPINGGTKMMLIB_LAYERMAPPINGAPPLET_H

