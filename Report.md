
[//]: # (Image References)


# Project 1: Navigation

### The environment

In this project, a model was trained to navigate a large square world to collect different bananas using a DQN.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Model Architecture

The DQN architecture implemented is shown below where fc_units = 512. The model receives information in the form of a vector of size 37 from the environment and outputs a vector of size 4 from its output layer containing values for each action.

```
self.fc1 = nn.Linear(state_size, fc_units)
self.fc2 = nn.Linear(fc_units, fc_units)
self.fc3 = nn.Linear(fc_units, fc_units)
self.fc4 = nn.Linear(fc_units, action_size)
```

### Hyperparameters

The hyperparameters are identical to a previous implementation and worked well for this agent.

```
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4
```

### Training

Below shows average score the agent achieved every 100 episodes. The increasing score value indicates the agent is learning about the environment and adapting their actions to maximise their score. The process took 572 episodes and a total training time of 14 minutes on the udacity workspace with GPU enabled to achieve an average score of 13.03.

Seeing that Udacity's example implementation solved the problem in around 1800 episodes, it was quite surpirising how out model finished training in just under 600 episodes. This is probably attributed to the additional hidden layer in the neural network and the increase of nodes for each hidden layer to 512 

```
Episode 100	Average Score: 0.21
Episode 200	Average Score: 2.22
Episode 300	Average Score: 5.70
Episode 400	Average Score: 9.38
Episode 500	Average Score: 11.48
Episode 572	Average Score: 13.03
Environment solved in 472 episodes!	Average Score: 13.03
```

![Screenshot 2021-03-05 at 10 30 37](https://user-images.githubusercontent.com/74315440/110096061-d4e44800-7d9d-11eb-9d32-edd912059548.png)

### Future Work

Possible future work could including: 
- training the agent with the pixels as input data (which would more closely mimic how we would play the game)
- Experimenting with policy based methods and comparing the two approaches.
