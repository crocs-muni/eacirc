#! /bin/bash

VERSION=1.0
LOG_FILE="batch.log"
EACIRC_BINARY="EACirc"
EACIRC_ARGS="-log2file"
RESULTS_FILES="config.xml eacirc.log scores.log population_initial.xml population.xml state_initial.xml state.xml avgfit_graph.txt bestfit_graph.txt fitness_progress.txt EAC_circuit.xml EAC_circuit.dot EAC_circuit.c EAC_circuit.txt"

echo "EACirc batch runner (version "$VERSION")"

if [ $# -eq 0 ]
then
	echo "usage: "$0" <config-file> [<config-file> ...]"
	echo
	echo " - runs EACirc with given configurations in sequence"
	echo " - currently set EACirc binary: "$EACIRC_BINARY
	echo " - saved results files: "$RESULTS_FILES
	echo " - results are saved into numbered folders"
	exit -1
fi

if [ ! -x $EACIRC_BINARY ]
then
	echo "error: "$EACIRC_BINARY" does not exist or is not executable."
	exit -1;
fi

DIR=0;
for CONFIG_FILE in "$@"
do
	DIR=$((DIR + 1))
	echo "=> processing file "$CONFIG_FILE" into dir "$DIR
	./$EACIRC_BINARY $EACIRC_ARGS -c $CONFIG_FILE
	if [ ! -d $DIR ]
	then
		mkdir $DIR
	fi
	cp $RESULTS_FILES $DIR
done
