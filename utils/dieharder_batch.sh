#! /bin/bash

VERSION=1.0
RESULTS_SUFFIX=".results"

# dieharder settings
TESTS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17"
PSAMPLES=1
GENERATOR=201
FLAGS_HEADER=511
FLAGS_NO_HEADER=504

echo "Dieharder batch script (version "$VERSION")"

if [ $# -eq 0 ]
then
	echo "usage: "$0" <binary-file> [<binary-file> ...]"
	echo
	echo "- runs Dieharder tests no. "$TESTS
	echo "- each test is run "$PSAMPLES" times (if possible)"
	echo "- random data is taken from file given as argument"
	echo "- results for each provided file are saved to text file with suffix "$RESULTS_SUFFIX
	echo "- dependecy: dieharder"
	exit -1
fi

echo "settings: psamples("$PSAMPLES"), tests ("$TESTS")"
for FILE in "$@"
do
	if [ ! -f $FILE ]
	then
		echo "File \""$FILE"\" does not exist!"
		continue
	fi
	echo "=> processing file "$FILE
	RESULTSFILE=$FILE$RESULTS_SUFFIX
	FIRST=1
	for TEST in $TESTS
	do
		if [ $FIRST -eq 1 ]
		then
			dieharder -d $TEST -p $PSAMPLES -D $FLAGS_HEADER -g $GENERATOR -f $FILE >$RESULTSFILE 2>&1
			FIRST=0
		else
			dieharder -d $TEST -p $PSAMPLES -D $FLAGS_NO_HEADER -g $GENERATOR -f $FILE >>$RESULTSFILE 2>&1
		fi
	done
	FAILED=`cat $RESULTSFILE | grep FAILED | wc -l`
	WEAK=`cat $RESULTSFILE | grep WEAK | wc -l`
	PASSED=`cat $RESULTSFILE | grep PASSED | wc -l`
	TOTAL=$(($FAILED + $PASSED + $WEAK))
	SCORE=`echo $PASSED" + 0.5 * "$WEAK | bc -l`
	echo "# number of tests total: "$TOTAL >>$RESULTSFILE
	echo "# number of FAILED:    "$FAILED >>$RESULTSFILE
	echo "# number of WEAKNESSES:  "$WEAK >>$RESULTSFILE
	echo "# number of PASSED:      "$PASSED >>$RESULTSFILE
	echo "# SCORE: "$SCORE"/"$TOTAL >>$RESULTSFILE
done
