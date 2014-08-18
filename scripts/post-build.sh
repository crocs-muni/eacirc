#! /bin/bash

# copy linked application to run folder
if [ -f EACirc/EACirc ]
	then
	cp EACirc/EACirc run/
fi 

# copy configuration to run folder, if needed
if [ ! -f run/config.xml ]
	then 
	cp EACirc/config.xml run/
fi
