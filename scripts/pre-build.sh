#! /bin/bash
(set -o igncr) 2>/dev/null && set -o igncr; # this comment is needed
# previous line tells bash to ignore '\r' (issue when running in WIN environments)

# prepare run folder
mkdir -p run

# EACirc pre build script setting current commit hash to Version.h
echo "// current git commit info (updated before build if correctly set-up)" >EACirc/Version.h

command -v git >/dev/null 2>&1
if [ $? -eq 0 ] 
then
	git log -1 --format="#define GIT_COMMIT \"%H\"" >>EACirc/Version.h
	git log -1 --format="#define GIT_COMMIT_SHORT \"%h\"" >>EACirc/Version.h
else
	echo "#define GIT_COMMIT \"n/a\"" >>EACirc/Version.h
	echo "#define GIT_COMMIT_SHORT \"n/a\"" >>EACirc/Version.h
fi
