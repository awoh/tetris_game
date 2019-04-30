# Tetris starter kit


## Setup instructions
First it is recommended to install conda: https://docs.conda.io/en/latest/miniconda.html
Then run
```
conda env create -f env.yml
```
to create conda environmnet `rltetris`

## Usage
```
python train_PI.py --debug --co
```

# Implementation Strategy

1. Fill in template, get working for small problem -- try Blackjack-v0. Then move on to a reduced form of tetris
1. Make sure to carefully decide what information you will need to log and track. Decide this up front and then make sure it is saved in your main training loop. It is critical to get this done first so that you can actually see what the updates are doing.
2. Slowly scale up, solve iteratively harder problems. Make sure to keep track of all results
2. Add additional environments and switch between them by passing command line args. DO NOT keep editing the same environment once it is working -- this will make it very hard to get the code working and to reproduce your results. Wherever possible, pass configuration via CLI arguments but do not change params in the code directly.
3. Implement plotting capabilities sooner rather than later (can use code from torchkit), this will help
check if stuff is working or not.


**NOTE:** A fixed random seed is set by default to aid debugging
**NOTE:** In order to quickly generate the samples needed to learn tetris you may need to run samplers in parallel. Be very careful that you avoid unnecessary overhead. This will require chunking the job in a smart way.

## Gym and Wrappers
For gym, if you want to modify the `(S,A,R)` tuple returned by the environment's `step` function, the best way is to create a wrapper. Literally, it just wraps the environment and modifies this tuple.

For example, for tetris, you might implement the environment such that it returns the complete board as the `S` in `(S,A,R)`. Then, you can write a wrapper to take this and return the different features -- eg Bertsekas features or DT features.


## Logging
Much better to use python logger than print statements
Fortunately, logger can be used as drop in replacement, so it is easy to switch
Then logs can be redirected to different output locations (eg logfile) when you run on a server.

just add
```
import logging
logger = logging.getLogger(__name__)
```
to the top of each file where you need to write logs.
