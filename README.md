# Introduction

## What is this repository?

This is a template docker-based dev environment. It currently supports NVIDIA GPUs but with slight modifications it can target for x86 CPUs and Apple silicon chips. 

It currently includes the following tools:

* an empty CLI tool that should be based on `typer`. 
* an empty library called `artagents`  where you can include your code / logic that you want to import across assignments and projects.
* a `notebooks` directory with an empty notebook where you need to populate with your code. The notebook can use the artagents library. 
* a `docs` directory that contains the source code of a `quarto` based publishing system based on markdown and `ipynb` notebooks. 
* a `tests` directory that should contains `pytest` based tests  for your `artagents` library and other code.

## What should I do with it?

* Clone it as you will need it to submit your work in this class. 
* Familiarize yourself with the `rye` package manager as you will use it it build the library and manage all your dependencies. 
* Follow the instructions in the course web site under resources to submit your repo to the course's LLM system (Canvas or Brightspace). 