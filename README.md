Multi-Agent Reinforcement Learning with Attention-Based Actor-Critic
Overview
This project implements a multi-agent reinforcement learning framework using an attention-based actor-critic model. It is designed to handle complex environments with multiple interacting agents, each learning its own policy while taking into account the state and actions of other agents.

Key Components
Environment: A custom environment class (Environment) that simulates the interactions between multiple agents.
Agent: Each agent is equipped with an attention-based actor (AttentionActor) and two critics (Critic) for value estimation.
Replay Memory: A memory buffer (ReplayMemory) to store transitions for experience replay.
Trainer: A class (Trainer) responsible for training the agents, including data sampling, model updating, and evaluation.


Installation
To run this project, you need to have the following libraries installed:
Python 3.8
PyTorch-cuda 12.1
NumPy 1.24.3
You can install these dependencies using pip:

Usage
Setup Environment:
Create an instance of the Environment class with the desired number of agents, environment dimensions, and evolution time step.
Initialize Agents:
Create a list of Agent instances, each configured with the necessary parameters.
Train the Agents:
Instantiate the Trainer class with the agents and the environment, then call the train method to start the training process.
Example Code
Here is a simplified version of the main training loop:

python
def main():
    n_agents = 3
    env_width = 3
    env_height = 3
    env_evo_dt = 0.1
 
    batch_size = 512
    max_episode = 100000000
 
    # Initialize the environment
    environment = Environment(n_agents, env_width, env_height, env_evo_dt)
 
    # Initialize the agents
    agents = [Agent(n_agents, environment.state_dim, environment.observe_dim, environment.action_dim, 128) for _ in range(n_agents)]
 
    # Initialize the trainer
    trainer = Trainer(agents, environment, gamma=0.99)
 
    # Start training
    trainer.train(batch_size, max_episode)
 
if __name__ == '__main__':
    main()
Configuration
State Dimension: The total number of features in the state space.
Observe Dimension: The number of features observed by each agent.
Action Dimension: The number of actions each agent can take.
Hidden Dimension: The size of the hidden layers in the actor and critic networks.
Gamma: The discount factor for future rewards.
Batch Size: The number of transitions sampled from the replay memory for each training step.
Max Episode: The maximum number of episodes to train for.
Training and Evaluation
Training: The train method runs the training loop, where agents explore the environment, store transitions in the replay memory, and periodically sample batches to update their policies and value functions.
Evaluation: The eval method allows for evaluating the trained agents in the environment without exploration noise. It also supports plotting the agents' trajectories as a GIF animation.
Results
Losses and rewards are tracked during training and saved to files for later analysis.
The trained models can be saved and loaded for further experimentation or deployment.
Contributions
Contributions to this project are welcome. Please follow the standard GitHub workflow of forking the repository, making changes, and submitting a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
