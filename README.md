README

These are the scripts to process EC data from raw data to fluxes. 

Processing_main.py is the main script where all the different functions are executed.
Func_read_data.py reads the raw data and puts it in a dataframe. 
Func_despike.py applies water vapor correction of the LICOR and a despiking algorithm according to Sigmund et al 2022.
Func_DR.py applies a double rotation to the fast dataframe.
Func_MRFD.py applies a Multi Resolution Flux Decomposition to find the correct averaging window.





