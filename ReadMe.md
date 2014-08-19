# EACirc Framework

[![Build status](https://travis-ci.org/petrs/EACirc.svg?branch=master)](https://travis-ci.org/petrs/EACirc)

EACirc is a framework for automatic problem solving. It uses supervised learning techniques based on evolutionary algorithms to construct and optimize software circuits in order to solve the given problem.

Problems are solved by the means of hardware-like circuits - small, software-emulated circuits consisting of gates and interconnecting wires transforming input data into desired output data. The layout of these circuits is designed randomly at first. They are subsequently optimized in the process of supervised learning (inputs are provided alongside with correct outputs) until the the desired success rate is achieved. 

The learning stage incorporates genetic programming principles:  
* a handful of these circuits (circuit 'population') is considered simultaneously;
* each individual circuit is evaluated on the data and its 'fitness' is determined by comparison of its outputs with the expected outputs;
* individuals with low 'fitness' are deleted (survival of the fittest);
* individuals with high 'fitness' are altered ('sexiual crossover' and a small chance of 'mutation');
* the process starts over with this new 'generation' of circuits.

## The Framework

The EACirc framework consists of main application and several supporting tools and scripts. The modular design allows for easy addition of new problem modules ('projects') and output interpretation modules ('evaluators'). Currently, the project has following main parts:
* **EACirc** - the main application, constructs circuits using evolutionary principles.
* **utils** - set of scripts and small programs used for results processing.

For more information and details see [projec wiki pages](http://github.com/petrs/EACirc/wiki/Home).

## Authors
The framework is developed at the [Laboratory of Security and Applied Cryptography](http://www.fi.muni.cz/research/laboratories/labak/), [Masaryk University](http://www.muni.cz/), Brno, Czech Republic.

* **Milan Čermák** 2012-now (CUDA support)
* **Karel Kubíček** 2014-now (TEA, CAESAR)
* **Zdenek Říha** 2013-now (bytecode emulator)
* **Marek Sýs** 2013-now (project concept, results interpretation)
* **Petr Švenda** 2008-now (project lead, initial implementation)
* **Martin Ukrop** 2012-now (framework model, refactoring, SHA-3 candidates testing, supporting tools)

Former participation:
* **Ondrej Dubovec** 2011-2012 (SHA-3 candidates testing)
* **Matěj Prišťák** 2011-2012 (object model and refactoring, XML support, eStream candidates testing)
* **Tobiáš Smolka** 2011-2012 (BOINC related support)