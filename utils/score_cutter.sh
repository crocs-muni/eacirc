#! /bin/bash

VERSION=1.0
SCORE_FILE="scores.log"
BACKUP_SUFFIX=".bak"

echo "Config cutter script (version "$VERSION")"

if [ $# -eq 0 ]
then
	echo "usage: "$0" <numGenerations>"
	echo 
	echo "- cuts score file in each subdirectory"
	echo "- only first <numGenerations> are left"
	echo "- original score files get backed up in suffix "$BACKUP_SUFFIX
	exit -1
fi

OLDPWD=`pwd`
echo "=> processing folder "$OLDPWD

for SUBDIR in *
do 
	if [ ! -d $SUBDIR ]
	then
		continue
	fi
	cd $SUBDIR
	if [ -f $SCORE_FILE ]
	then
		cp $SCORE_FILE $SCORE_FILE$BACKUP_SUFFIX
		head -$1 $SCORE_FILE$BACKUP_SUFFIX >$SCORE_FILE
	fi
	cd $OLDPWD
done
