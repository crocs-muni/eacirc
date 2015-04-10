#! /bin/bash
(set -o igncr) 2>/dev/null && set -o igncr; # this comment is needed
# previous line tells bash to ignore '\r' (issue when running in WIN environments)

# remove all results files
cd run
rm -f *.log *.txt *.bin *.c *.dot *.xml *.2
rm -f EACirc
