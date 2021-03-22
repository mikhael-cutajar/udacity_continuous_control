[//]: # (Image References)


# Project 2: Continuous Control
### The environment

In this project, a model using DDPG was trained to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## The code

The code was implemented by adjusting [Udacity's ddpg pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) to work for the continuous control environment with 20 agents.

### Model Architecture

The ddpg model architecture implemented was similar to the one used in the [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf) where we had two hidden layers with 400 and 300 units respectively. As opposed to the Udacity pendulum implementation we used batch normalisation on both the actor and the critic. 

### Hyperparameters

The hyperparameters are identical to the Udacity pendulum implementation as they worked well. The only change made which greatly improved results was the increase of max_t to 1300.

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

### Training

Below shows the score and average score the agent achieved every 10 episodes. The increasing score value indicates the agent is learning about the environment and adapting their actions to maximise their score. The process took 80 episodes running on the udacity workspace with GPU enabled to achieve an average score of 30.21.

At first the model was refusing to learn never surpassing a score of 10 after several episodes, however the changes to the model architecture and the increase to max_t was sufficient to help the model beat the game.

```
Episode 10	Score: 1.56	Average Score: 0.58
Episode 20	Score: 2.87	Average Score: 1.93
Episode 30	Score: 5.79	Average Score: 4.89
Episode 40	Score: 7.76	Average Score: 6.92
Episode 50	Score: 12.94	Average Score: 10.41
Episode 60	Score: 15.86	Average Score: 14.69
Episode 70	Score: 21.09	Average Score: 18.36
Episode 80	Score: 35.86	Average Score: 30.21

Environment solved by episode 80
```

![Screenshot 2021-03-18 at 17 12 29](https://user-images.githubusercontent.com/74315440/111659070-33351000-880d-11eb-967a-7fea5f52fbca.png)

### Future Work

Possible future work could including: 
- Trying out the D4PG model and comparing the results
- Experiment with how learning changes when using ven more agents
