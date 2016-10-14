![EACirc](https://raw.githubusercontent.com/wiki/petrs/EACirc/img/logo-home.png)  
[![Build Status](https://travis-ci.org/crocs-muni/eacirc.svg?branch=master)](https://travis-ci.org/crocs-muni/eacirc) [![Coverity status](https://scan.coverity.com/projects/7192/badge.svg)](https://scan.coverity.com/projects/crocs-muni-eacirc)
[![Latest release](https://img.shields.io/github/release/crocs-muni/EACirc.svg)](https://github.com/crocs-muni/EACirc/releases/latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.160808.svg)](https://doi.org/10.5281/zenodo.160808)

EACirc is a framework for automatic problem solving. It can be utilized as randomness testing tool similar to statistical batteries (NIST STS, Dieaharder, TestU01), for instance for analysis of cryptographic function outputs.

It uses supervised learning techniques based on metaheuristics to construct adapted distinguisher of two input data streams. The distinguisher can be represented as hardware-like circuits or algebraic polynomial. 

## The Framework

This repository contains EACirc core and code for data stream generation (mainly eSTREAM and SHA-3 candidates).
Further tools are:
* [Randomness Testing Toolkit (RTT)](https://github.com/crocs-muni/randomness-testing-toolkit),
* [tools for GRID computations](https://github.com/crocs-muni/eacirc-utils)
* [Oneclick](https://github.com/crocs-muni/oneclick) a tool for BOINC computation (deprecated)

For more information and details see [project wiki pages](http://github.com/petrs/EACirc/wiki/Home).

## Authors
The framework is developed at the [Centre for Research on Cryptography and Security (formerly Laboratory of Security and Applied Cryptography)](https://www.fi.muni.cz/research/crocs/), [Masaryk University](http://www.muni.cz/), Brno, Czech Republic.

* **Petr Švenda** 2008-now (project lead, initial implementation)
* **Jiří Novotný** 2014-now (build system, CUDA, main developer)
* **Michal Hajas** 2015-now (Java bytecode emulator)
* **Dušan Klinec** 2012-now (polynomial distinguisher)
* **Karel Kubíček** 2014-now (TEA, metaheuristics)
* **Ľubomír Obrátil** 2014-now (RTT, Oneclick)
* **Marek Sýs** 2013-now (statistics evaluation, polynomials)
* **Martin Ukrop** 2012-now (framework model, refactoring, SHA-3 & CAESAR candidates testing, supporting tools)

Former participation:
* **Milan Čermák** 2012-2013 (CUDA)
* **Ondrej Dubovec** 2011-2012 (SHA-3 candidates testing)
* **Matěj Prišťák** 2011-2012 (object model and refactoring, XML support, eStream candidates testing)
* **Zdenek Říha** 2013-2016 (Java bytecode emulator)
* **Tobiáš Smolka** 2011-2012 (BOINC related support)

 
