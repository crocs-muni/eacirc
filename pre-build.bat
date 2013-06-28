@echo off
where git
if errorlevel 1 (goto nok) else goto ok

:nok
echo // current git commit info (updated before build if correctly set-up) >EACirc/Version.h &
echo #define GIT_COMMIT "n/a">>EACirc/Version.h &
echo #define GIT_COMMIT_SHORT "n/a">>EACirc/Version.h
goto end

:ok
echo // current git commit info (updated before build if correctly set-up) >EACirc/Version.h
git log -1 --format="#define GIT_COMMIT \"%%H\"" >>EACirc/Version.h
git log -1 --format="#define GIT_COMMIT_SHORT \"%%h\"" >>EACirc/Version.h
goto end

:end