#ifdef WANTROOT

#include <TSystem.h>
#include <TRandom.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <thread>
#include <chrono>
#include <iostream>


int main ( int argc, char* argv[] )
{
    TApplication app ("app",&argc,argv);
    TCanvas *canvas = new TCanvas("fCanvas", "fCanvas", 600, 400);
    TH1D h ("h","h",10,0,10);
    h.Fill(1);
    h.Draw();
    canvas->Modified();
    canvas->Update();
    while (!gSystem->ProcessEvents()) {
       h.Fill(10 * gRandom->Rndm());
       canvas->Modified();
       canvas->Update();
       gSystem->Sleep(100);
    } 
    app.Run();
    return 0;
}


/*
int main ( int argc, char* argv[] )

{
    TApplication app ("app",&argc,argv);
    TCanvas *canvas = new TCanvas("fCanvas", "fCanvas", 600, 400);
    TH1D h ("h","h",10,0,10);

    h.Fill(1);
    h.Fill(2);
    h.Fill(2);
    h.Fill(2);
    h.Fill(3);
    h.Fill(3);
    
    h.Draw();
    canvas->Update();
    canvas->Draw();

  
    gSystem->ProcessEvents();

    std::this_thread::sleep_for( std::chrono::seconds(3) );

    h.Fill(4);
    h.Fill(5);
    h.Fill(5);
    h.Fill(6);
    h.Fill(6);
    h.Fill(6);
    
    h.Draw();   
    canvas->Update();
    canvas->Draw();

 
    gSystem->ProcessEvents();

    std::this_thread::sleep_for( std::chrono::seconds(3) );

    return 0;

}
*/

#else 

#include <iostream>

int main(int argc, char* argv[])
{
    std::cout << "ROOT has been disabled in this build so this test cannot be performed. Exiting.\n";
    return 0;
}

#endif

