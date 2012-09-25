# EACirc Project

Framework for automatic search for circuit problem solving via Evolutionary algorithms

## Building instructions

Application can be comfortably build in 3 independent ways: using MS Visual Studio, using QT Creator or using GNU Make. If using other methods, please make sure to add three additional paths to includepath: EACirc/ EACirc/galib EACirc/tinyXML

### MS Visual Studio:

(?)

### QT Creator:

G++-4.7 or higher required (or equivalent, e.g. MinGW-4.7 or higher). Load the QTProject file provided with the code. Instructions for building and linking libraries (GAlib and tinyXML) externally are provided in the project file.  
Note: to run the application, config.xml must be present in the running folder (determined by setting in Projects > Run Settings > Run > Working directory)

For running comfortably from within the Qt Creator, separate run folder can be used and custom build and clean steps set to handle config.xml, see below for examples.

_Note:_ the "run in terminal" option must be disabled in som configurations (especially when debugging)

* Windows:  
    in project run settings edit the working directory (e.g. to "%{buildDir}/run")  
    create 2 batch files in build directory (e.g. post-build.bat and post-clean.bat) with this contents  
    post-build.bat: copy EACirc\config.xml run\config.xml  
    post-clean.bat: cd run && del *.txt *.dot *.bin *.c scores.log config.xml  
    add cutom build step: post-build.bat (command) in %{buildDir} (working dir)   
    custom clean step: post-clean.bat (command) in %{buildDir} (working dir)
* Linux:  
    in project run settings edit the working directory (e.g. to "%{buildDir}/../run")  
    add cutom build step: cp (command) ../EACirc/config.xml config.xml (arguments) in %{buildDir}/../run (working dir)  
    if planning to run from cli, add cutom build step: cp (command) EACirc %{buildDir}/../run (arguments) in %{buildDir} (working dir)  
    custom clean step: rm (command) -f *.txt *.dot *.bin *.c scores.log config.xml (arguments) in %{buildDir}/../run (working dir)

### GNU Make

EACirc makefile can be used on UNIX platforms only (due to shell commands and paths format), however libraries GAlib and tinyXML can be build on other platforms using equivalent of make (e.g. mingw32-make). Details provided in the makefile.

EACirc requires G++-4.7 or higher. Before running 'make -f EACirc.makefile <target>' chceck the settings at the top of the 'makefile'.  
There are several targets for make:

* all        (builds libraries, application and links everything together)
* libs       (builds both GAlib and tinyXML libraries)
* main       (builds and links the main application - libraries need to be built before!)
* clean      (cleans the previous build, both libraries and main application)
* cleanmain  (cleans the previous build, main application only)

## Running instructions

* To run the application, set the config.xml file and run the binary without any parameter.
* If you want to start with partly evolved circuit, put the circuit in .bin format into the executable folder.
* To disable evolution, run the binary with "-evolutionoff" parameter.
* To test the static circuit, run the binary with "-staticcircuit" parameter (in this case, you will need pre-generated test vectors as binary files and pre-compiled evolved circuit).
