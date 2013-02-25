#! /bin/bash

VERSION=1.0
RESULTS_FILE="errors.log"
CONFIG_FILE="config.xml"

echo "Config checker batch script (version "$VERSION")"

if [ $# -eq 0 ]
then
	echo "usage: "$0" <experiment-folder> [<experiment-folder> ...]"
	echo
	echo "- checks whether configuration files ("$CONFIG_FILE") are the same for all runs"
	echo "- result log is written to "$RESULTS_FILE" in experiment directory"
	echo "- <experiment-folder> must end with path separator"
	exit -1
fi

OLDPWD=`pwd`
for FILE in "$@"
do
	cd $OLDPWD
	if [ ! -d $FILE ]
	then
		echo "File \""$FILE"\" does not exist or is not a directory!"
		continue
	fi
	echo "=> processing experiment "$FILE
	FOUND=0
	cd $FILE
	rm $RESULTS_FILE
	for RUN in *
	do
		if [ ! -d $RUN ]
		then
			continue
		fi
		DIFFERENCES=		
		for RUN2 in *
		do
			if [ ! -d $RUN2 -o $RUN = $RUN2 ]
			then
				continue
			fi
			DIFF_COUNT=0
			DIFF_COUNT=`diff $RUN"/"$CONFIG_FILE $RUN2"/"$CONFIG_FILE | wc -l`
			if [ $DIFF_COUNT -ne 0 ]
			then
				DIFFERENCES=$DIFFERENCES" "$RUN2"["$DIFF_COUNT"]"
				FOUND=1
			fi
		done
		if [ ! -z "$DIFFERENCES" ]
		then
			echo $CONFIG_FILE" in "$RUN" differs from runs "$DIFFERENCES >>$RESULTS_FILE
		fi
	done
	if [ $FOUND -eq 0 ]
	then
		echo "No differences found." >>$RESULTS_FILE
	fi
done
cd $OLDPWD
