REM Fill your workunit name (e.g., Decim8_run_categ10_1000chf_50kgener_fixseedid) and number of parallel workunits started (e.g., 30)

powershell.exe .\get_boinc_result_scores.ps1 'Decim8_run_categ10_1000chf_fixseedid' 10 > start.bat
start.bat
