=== EACirc Project ===

== Building instructions ==

Application can be build in 3 independent ways: using MS Visual Studio, using QT Creator or using GNU Make.
 * MS Visual Studio:
    (?)
 * QT Creator:
    G++-4.7 or higher required (or equivalent, e.g. MinGW-4.7 or higher). Load the QTProject file provided with the code. Instructions for building and linking libraries (GAlib and tinyXML) externally are provided in the project file.
 * GNU Make
    EACirc makefile can be used on UNIX platforms only (due to shell commands and paths format), however libraries GAlib and tinyXML can be build on other platforms using equivalent of make (e.g. mingw32-make). Details provided in the makefile.
    EACirc requires G++-4.7 or higher. Before running 'make' chceck the settings at the top of the 'makefile'. There are several targets for make:
        make all        (builds libraries, application and links everything together)
        make libs       (builds both GAlib and tinyXML libraries)
        make main       (builds and links the main application - libraries need to be built before!)
        make clean      (cleans the previous build, both libraries and main application)
        make cleanmain  (cleans the previous build, main application only)

== Running instructions ==

To run the application, set the config.xml file and run the binary without any parameter.
If you want to start with partly evolved circuit, put the circuit in .bin format into the executable folder.
To disable evolution, run the binary with "-evolutionoff" parameter.
To test the static circuit, run the binary with "-staticcircuit" parameter (in this case, you will need pre-generated test vectors as binary files and pre-compiled evolved circuit).
