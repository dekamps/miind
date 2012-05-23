//------------------User defintions---------------------

// GridGrab selects the grid from the file which belongs to the
// selected population, at the closest time available. Since, the
// simulation results may not be produced exactly at the desired time,
// the simulation time of the selected grid is produced for comparison.
// If no grids could be found which match the selection criteria, false is
// returned, otherwise true. 

bool StateGridGrab
(
     TFile*,   // pointer to an open file
     Int_t,    // which population
     Float_t,  // which time should be selected
     Float_t*, // actual time selected
     TGraph**  // the desired graph
);



//------------------Auxilliary definitions--------------

const Double_t LARGE = 100000;

bool ParseGraphName
(
     char*  p_name,
     int*   p_n,
     float* p_time
)
{
     string name(p_name);
 
     if ( name.find("grid") == string::npos )
        return false;

     int i1 = name.find("_");
    
     int i2 = name.rfind("_");
     
     string num;
     // can get stl copy to work
     for (int i = i1 + 1; i < i2; i++)
       num.append(name, i1 + 1, i2 - i1 - 1);

     *p_n = atoi(num.c_str());

     char buffer[10000];
     strcpy(buffer,name.c_str() +i2 + 1);

     string time(buffer);
     *p_time = atof(buffer);
}

bool StateGridGrab
(
     TFile*   p_file,
     Int_t    n_grid,
     Float_t  time_desired,
     Float_t* p_time_real,
     TGraph**  p_state_graph
)
{
      Float_t diff_before = LARGE;

      // loop over all TKey's in file
      // essentially a loop over all object names
     
      TIter nextkey(p_file->GetListOfKeys());
      TKey* key; 
      TKey* old;
      while (key = (TKey*)nextkey()) 
      {
	   Float_t  time_grid;      // simulation time of the current grid
	   Int_t    n_name_grid;    // n of the current grid, to be compared to the desired grid

	   bool b_test = ParseGraphName(key->GetName(),&n_name_grid, &time_grid);
       
   
	   // if we have the correct grid, 
	   if ( n_grid == n_name_grid )
	   {
	       Double_t act_diff =  fabs(time_desired - time_grid);
       	       if (act_diff < diff_before)
	       {
	            diff_before = act_diff;
               }
               else
	      {
                   // ok, we are past the right index, get the old one
	           // don't test for zero: the first key can't be the one
	  
                   *p_state_graph = (TGraph*)old->ReadObj();
                   ParseGraphName(old->GetName(), &n_name_grid, &time_grid);
                   *p_time_real = time_grid;

		   break;
	      }
              // you want the last grid of the same kind
	      old = key;
	   }
       }

       return true;
}     
