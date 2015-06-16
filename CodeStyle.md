# Code style for EACirc

To maintain codebase consistency, please adhere to these guidelines when contributing to EACirc or any other supporting tools. Most importatntly, **retain style consitency**, i.e. try to use the same conventions as you see in the code.

Some specific notes and guidelines are below.

## Code design

* **Check configuration sanity, report and fail gracefully**
  If your code uses several settings, try to check for the incompatible combinations at the beginning. If found, report the failure (as precisely as possible) and fail the program gracefully (try to deallocate memory and exit). Do not try to guess the correct settings the user wanted.
* **Avoid C-style casting**
  Choose the appropriate C++ casting method: static_cast, dynamic_cast, const_cast or reinterpret_cast. Try to avoid reinterpret_cast.
* **Inspect status codes**
  If you call a method returning status code, inspect it. In case of failure, propagate above. If your method can fail, return the status code.

## Documentation

* **Keep developer Wiki up-to-date**
  After significat changes or feature addition, try to document these in the developer wiki (At least create the TBA sections).
* **Document methods**
  Write a short JavaDoc-style documentation for all methods, mention any restriction on the parameters (maximum supported length, etc.).

## Fromating

* **Avoid tabs**
  Use spaces for indentation, indent is 4 characters wide. Try not to produce trailing spaces.

## Experiments

* **Retain data**
  For any notable experiments, try to retain the data allowing for replicating the runs. The most important is the config file and log file (includes date, time and commit).
* **Note your (non-)discoveries**
  Write the performed experiments into internal wiki. Even if you found out nothing (it is important to have notes abount performed experiments).
