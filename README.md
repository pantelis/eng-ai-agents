# Introduction

## What is this repository?

This is a template docker-based dev environment. It currently supports NVIDIA GPUs but with slight modifications it can target for x86 CPUs and Apple silicon chips. 

It currently includes the following tools:

* an empty library called `artagents`  where you can include your code / logic that you want to import across assignments and projects.
* a `assignments` directory with an empty notebook where you need to populate with your code. The notebook can optionally use the artagents library. 
* a `project` directory for your project source code. The documentation for the project is stored separately in the `docs` directory. 
* a `docs` directory that contains the source code of a [quarto](https://quarto.org/) based publishing system with markdown (qmd) and `ipynb` notebooks content. You use the docs folder to publish your project work. 
* an empty CLI tool that should be based on `typer` and you can optionally use to implement a CLI for your project or assignments. 
* a `tests` directory that should contains `pytest` based tests  for your `artagents` library and all other code.

## What should I do with it?

* Follow all instructions under [resources in the class website](https://pantelis.github.io/aiml-common/resources/environment/) as you will need it to submit your work. 
* Familiarize yourself with the `rye` package manager as you will use it it build the library and manage all your dependencies. 
* Follow the instructions in the course web site under resources to [submit your repo to the course's LLM system](https://pantelis.github.io/aiml-common/resources/environment/assignment-submission.html) (Canvas/Brightspace). 