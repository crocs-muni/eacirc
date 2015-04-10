:: requires 1 argument, current subdirectory in .\build to use
:: usually "debug", "release", "Debug", "Release" or "."
COPY .\build\%1\EACirc.exe .\run\EACirc.exe
COPY .\EACirc\config.xml .\run\config.xml