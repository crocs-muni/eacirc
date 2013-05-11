#! /bin/bash

VERSION=1.6
PARENTDIR=../../boinc/_processed
BASEHTML=http://centaur.fi.muni.cz:8000
HXCLEAN=hxclean

echo "Results downloader for LaBAK@BOINC, version "$VERSION
if [ $# -eq 0 ]
then 
	echo "usage: "$0" <results-html-file> [<results-html-file> ...]"
	echo
	echo " - provide one or more html files from LaBAK@BOINC/results"
	echo " - downloads all files provided in BOINC results table"
	echo " - only files for successfull runs are downloaded"
	echo " - downloaded jobs are storer in "$PARENTDIR" (this can be changes in script header)"
	echo " - dependency: html-xml-utils, currently called as '"$HXCLEAN"' (changed by constant)"
	exit -1
fi
if [ ! -d $PARENTDIR ]
then
	mkdir -p $PARENTDIR
fi

# main loop for given source files
for SOURCEHTML in "$@"
do
	if [ ! -f $SOURCEHTML ]
	then
		echo "File \""$SOURCEHTML"\" does not exist."
		continue		
	fi
	echo "Downloading results from LaBAK@BOINC for source html: "$SOURCEHTML
	CLEAN=$SOURCEHTML.clean
	# clean HTML
	$HXCLEAN $SOURCEHTML >$CLEAN
	ITEM=0
	while true 
	do
		ITEM=$[$ITEM + 1]
		DIR=`cat $CLEAN | hxselect .line-ok:nth-child\($ITEM\) td:nth-child\(2\) a | sed ':a;N;$!ba;s/[\r]\n/ /g'`
		DIR=`echo $DIR | sed 's/[^>]*>[ \t]*\([^<> \t]*\)[ \t]*<\/[^<]*/\1/' | sed 's/\([^[]*\)\[\(.*\)\]/\1\/\2/'`
		if [ -z "$DIR" ]; then break; fi
		DIR=$PARENTDIR/$DIR
		echo =\> Processing $DIR
		mkdir -p $DIR
		ROW=`cat $CLEAN | hxselect .line-ok:nth-child\($ITEM\) td:nth-child\(13\) a`
		FILES=`echo $ROW | sed 's/<a[^>]*>\([^<>]*\)<\/a>/\1 /g'`
		for FILE in $FILES
		do
			# echo $FILE
			SED="s/.*<a\([^>]*\)>"$FILE"<\/a>.*/\1/"
			URL1=`echo $ROW | sed $SED`
			URL2=`echo $URL1 | sed 's/[^"]*"\([^"]*\)".*/\1/'`
			# echo $BASEHTML$URL2
			# echo $DIR/$FILE
			# echo
			wget -q $BASEHTML$URL2 -O$DIR/$FILE
		done
	done
	rm $CLEAN
done
