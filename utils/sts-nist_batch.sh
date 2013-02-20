#! /bin/bash

VERSION=1.0
RESULTS_SUFFIX=".sts-nist"
LOG_SUFFIX=$RESULTS_SUFFIX".log"
STS_PATH="../../sts-2.1.1"
STS_BINARY="assess"

# STS NIST settings
STREAM_SIZE=1000000
STREAM_NUMBER=100
STS_RESULTS_FILE="./experiments/AlgorithmTesting/finalAnalysisReport.txt"
STS_LOG_OK="correct-log.txt"

echo "STS-NIST batch script (version "$VERSION")"

if [ $# -eq 0 ]
then
	echo "usage: "$0" <binary-file> [<binary-file> ...]"
	echo
	echo "- runs STS NIST independently for each input file"
	echo "- stream size used: "$STREAM_SIZE
	echo "- number of streams for each test: "$STREAM_NUMBER
	echo "- results for each provided file are save to text file with suffix "$RESULTS_SUFFIX
	echo "- run logs are saved (if necessary) to text file with suffix "$LOG_SUFFIX
	echo "- dependency: sts-nist binary of name \""$STS_BINARY"\" at path "$STS_PATH
	exit -1;
fi

echo "settings: streamsize("$STREAM_SIZE"), number of streams("$STREAM_NUMBER")"
RUNDIR=`pwd`
cd $STS_PATH
for ARG in "$@"
do 
	if [ ! $(echo $ARG | sed 's|^/|xx/|') = "xx"$ARG ]
	then 
		# $ARG is not absolute path
		FILE=$RUNDIR"/"$ARG
	else
		FILE=$ARG
	fi
	if [ ! -f $FILE ]
	then
		echo "File \""$FILE"\" does not exist!"
		continue
	fi
	echo "=> processing file "$FILE
	RESULTSFILE=$FILE$RESULTS_SUFFIX
	LOGFILE=$FILE$LOG_SUFFIX
	CMD="echo 0 "$FILE" 1 0 "$STREAM_NUMBER" 1 | ./"$STS_BINARY" "$STREAM_SIZE
	# echo $CMD
	eval $CMD >$LOGFILE 
	if [ -f $STS_RESULTS_FILE ]
	then
		mv $STS_RESULTS_FILE $RESULTSFILE
	fi
	if [ -f $LOGFILE ]
	then
		DIFF=`diff $STS_LOG_OK $LOGFILE`
		if [ -z "$DIFF" ]
		then
			rm $LOGFILE
		else
		echo "Error detected when processing file \""$FILE"\". See log \""$LOGFILE"\"."
		fi
	else
		echo "Log file \""$LOGFILE"\" does not exist!"
	fi
done
cd $RUNDIR
