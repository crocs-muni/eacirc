#! /bin/bash

# TODO vor version 2
# Check error codes
# divide results files between optional and compulsory
# add log reporter after EACirc run (grep warning, error)

VERSION=1.9
RUN_FOLDER="run"
EACIRC_BINARY="eacirc-059188e-ssl-nocuda.exe"
EACIRC_ARGS=""
RESULTS_FILES="config.xml eacirc.log scores.log population_initial.xml population.xml state_initial.xml state.xml fitness_progress.txt EAC_circuit.xml EAC_circuit.dot EAC_circuit.c EAC_circuit.txt caesar_stream.bin estream_stream1.bin sha3_stream1.bin"
                
echo "EACirc batch runner (version "$VERSION")"

if [ $# -eq 0 ]
then
	echo "usage: "$0" <config-file> [<config-file> ...]"
	echo
	echo " - runs EACirc with given configurations in sequence"
	echo " - currently set EACirc binary: "$EACIRC_BINARY
  echo " - used CLI EACirc arguments: "$EACIRC_ARGS
	echo " - saved results files: "$RESULTS_FILES
	echo " - results are saved into .d folders according to config names"
	exit -1
fi

echo -n "Checking for EACirc executable... "
if [ ! -x $EACIRC_BINARY ]
then
	echo "ERROR\nerror: "$EACIRC_BINARY" does not exist or is not executable."
	exit -1;
fi
echo "OK"
               
for ARGUMENT in "$@"
do
  CONFIG_FILE=`basename $ARGUMENT`
  echo "Processing "$CONFIG_FILE":"
	RESULTS_DIR=$CONFIG_FILE.d
  rm -rf $RUN_FOLDER
  echo -n "  Copying files: "
  mkdir $RUN_FOLDER
  echo -n "executable, "
  cp $EACIRC_BINARY $RUN_FOLDER
  echo -n "config... "
  cp $ARGUMENT $RUN_FOLDER
  echo "OK"
	echo -n "  Running EACirc... "
  cd $RUN_FOLDER
	./$EACIRC_BINARY $EACIRC_ARGS -c $CONFIG_FILE 2>/dev/null
  echo "OK (warnings/errors below)"
  cat eacirc.log | grep error
  cat eacirc.log | grep warning                             
  echo -n "  Copying results out... "
	if [ ! -d ../$RESULTS_DIR ]
	then
		mkdir ../$RESULTS_DIR
	fi
	cp $RESULTS_FILES ../$RESULTS_DIR 2>/dev/null
  cd ..
  echo "OK"         
done
