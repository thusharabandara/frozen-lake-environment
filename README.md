# Frozen Lake Environment

This repository contains the code and data for the frozen lake experiment, which is conducted as part of the course: CO 541 Artificial Intelligence at the Department of Computer Engineering, University of Peradeniya, Sri Lanka.

## Background

The objective of this experiment was to investigate the performance of different reinforcement learning algorithms on the Frozen Lake environment, which is a popular benchmark environment in the field of reinforcement learning. The Frozen Lake environment is a grid-world game in which an agent moves from one grid cell to another, while trying to avoid falling into holes and reach the goal cell. The agent receives a reward of +1 for reaching the goal cell and a reward of 0 for all other actions.

## Methodology

In this experiment, the Q-learning algorithm is used. The training process of the algorithm is conducted on the Frozen Lake environment using different parameter settings and recorded their performance in terms of the number of episodes required to reach the goal state and the total rewards obtained.

## Repository Structure

This repository contains the following files:

- `frozen_lake.py`: Python script for implementing the reinforcement learning algorithms and running the experiment. Students must implement the given task to complete the algorithm.
- `.gitignore`: This file is used to specify which files and directories Git should ignore when committing changes to a repository. 
- `LICENSE`: This file contains the license terms under which this repository is distributed. It specifies the permissions and restrictions that others have when using, modifying, or distributing the codebase.

## Usage

To run the experiment, simply run the `frozen_lake.py` script in a Python environment with the necessary Python libraries (a.k.a. dependencies) installed. The script will train the algorithms on the Frozen Lake environment and display outputs, including figures for learning curves.

## Conclusion

Overall, this experiment provides valuable insights into the performance of different reinforcement learning algorithms on the Frozen Lake environment. The results can be used to guide future research in the field of reinforcement learning and provide a baseline for comparing the performance of new algorithms.