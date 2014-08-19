#! /bin/bash
(set -o igncr) 2>/dev/null && set -o igncr; # this comment is needed
# previous line tells bash to ignore '\r' (issue when running in WIN environments)

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
