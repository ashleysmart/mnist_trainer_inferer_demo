# Design choices

This sorts out the requirements and document my core design choices and the reason why i made them

## Documentation

- This document doc/Design.md... documents why decisions where made
- The Readme.md documents how to use the code
- The doc/Spec.md documents the original specs
- The doc/Research.md documents the core research that helped the design
- The code
    - inline comments:
        - when the code is unclear
        - when the intent is not clear
    - give the size of this project my comments are lighter then they should be
    - i assume u can run an ai assistant to explain the rest

** If you cant read code your not going to last as a programer as AI assistants and vibe coding grows in use **

## Defaults and hard setup

- Use a .env file to set defaults and allow commandline parameters to over ride them.
- Later teams that fork repo the over write the .env to change the defaults without need to hack into the main.py code

## Modular requirements

- limit the design to be for a mnist classification network.. otherwise this will be crazy large
- Create reusable/replaceable modules that work for:
    - datasets
    - models
    - training criterion or loss functions
    - optimizers
    - model exporters/savers
    - acceptance criteria
    - visualizations
- Each of the these can be selected via a factory function:
    - each factory function would be named `make_...`
    - for example
        - src/models/__init__.py:  `def make_model`
        - src/exporter.py: `def make_exporter`

## Training pipeline

- Limit the implementation to the training of classification networks
- The technical requirements are:
    - ubuntu
    - python
    - pytorch
    - logger
    - classifier
        - mnist data suggests it one..
        - auto-encoder is possible.. but its alot for a into test
- Build a docker image that has all of the requirements pre-installed
    - allows portability and reproduction
    - allows me to isolate the code from my personal system
    - use a gpu pass through in docker to get gpu access
- TODO
    - THIS NEEDS early stopping so it doesnt over train
    - THIS NEEDS best model saving in case the model starts to overfit
    - THIS NEEDS a training seed

## Evaluation

- give that this has a release requirement it requires both a validation and testing dataset
    - validation set to monitor generalization while training
    - test set to check release status
- metrics need
    - stored and machine readable, so dump a json file to keep it
    - present in the log, so the user can review and approve
- release decision
    - needs to be based on some criteria
        - min accuracy
        - min per class accuracy
        - max inference time
            - subject to deviation based on build machine..
        - max model size
            - a little frivolous.. but im out of ideas

## Logging

- logging uses python logger
- logging needs to be to files and direct output
- log the
    - metrics
    - evaluation
    - release approval

## CI/CD and Release

TODO -  IMCOMPLETE.. i would like this to be deployable as nvidia trition packages

NOTE:
- I DONT HAVE A NVIDIA TRITION ACCOUNT setup
    - I would deploy to a trition model repo! it the better sane choice
- the only system i have in my personal workspace is github.
    - my options are very limited
- implement a github CI/CD
    - this allows the github runners to build the release package
    - this allows the github runners and release packages mechanism to own and manage the release
- the CI/CD system should stop if a PR is not "reviewed",
    - approval of the PR to main is paramout to release a main and will trigger the package release
