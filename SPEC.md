This is a RL playground project to explore the MiniGrid RL environment and try out different RL methods on it.
It has a complete training, evaluation and demonstration pipeline, and let the user to experience, compare and intuitively understand different MiniGrid environments and RL methods better.

Observability, extendability and reproducibility are principal concerns of the framework design.

## RL Related

- We focus on MiniGrid. The observation space is the 7 * 7 * 3 uint8 image only. The action space is Discrete(7). This applies everywhere.
- For agents, it should accept the uint8 input, and output the int action directly. It's the agents' internal responsibility to decide how to convert.

## Requirements

The whole playground project should be working on both Mac and Linux.
It should automatically detect the existence of mps or cuda backend and use them if possible.
Pay attention to paralellism. Training efficiency is important.


## Training & Evaluation

The evaluation should be traced by tensorboard.


## Demonstration

The project provides a visualization frontend. 
The frontend provides the map, the agent observation, and the statistics like step number, accumulated reward, etc.
It serves two functionalities:
1. Let the user play the MiniGrid game with the keyboard.
2. Let the agent play the MiniGrid game.
3. Replay a recorded trace.

Notice that we use a unified adapter for the user and agent. 
The format of recorded trace should be the same. I propose it to be the sequence of actions.