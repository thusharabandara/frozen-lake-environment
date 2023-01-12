import gym
import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(1234)

class Agent:
    '''Q-learning agent'''

    def __init__(self, 
                 action_space, 
                 q_table_shape, 
                 alpha, gamma, 
                 epsilon):
        self.action_space = action_space
        self.q_table = np.zeros(shape=q_table_shape)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, state):
        '''Select action for exploration or exploitation'''
        # YOUR CODE HERE [1]
        return action

    def learn(self, state, action, reward, next_state):
        '''Update Q table based on experience'''
        # YOUR CODE HERE [2]
        self.q_table[state, action] = ... # new_value

# Run Q-learning experiment with FrozenLake
env = gym.make("FrozenLake-v1")
# env.seed(1234)
num_episodes = 10000
max_steps = 100
num_runs = 5
total_reward = np.zeros(shape=(num_runs,))
acc_reward = np.zeros(shape=(num_runs, num_episodes))
alpha = 0.1
gamma = 0.99
epsilon = 1000
min_epsilon = 10

# Run experiment num_runs times
for i in range(num_runs):
    print("Run: ", i+1)
    my_agent = Agent(env.action_space, 
                 (env.observation_space.n, env.action_space.n), 
                 alpha, gamma, epsilon)
    # Run training for num_episodes
    for j in range(num_episodes):
        done = False
        t = 0
        ep_reward = 0
        state = env.reset()
    
        while t < max_steps:
            action = my_agent.act(state)
            next_state, reward, done, info = env.step(action)
            my_agent.learn(state, action, reward, next_state)
            state = next_state
            t += 1
            ep_reward += reward
            if done:
                break

        # Update my_agent.epsilon according to chosen strategy for exploration
        # YOUR CODE HERE [5]
        my_agent.epsilon = ... # epsilon value

        total_reward[i] += ep_reward
        acc_reward[i, j] = total_reward[i]

print("Mean episode reward: ", np.sum(total_reward)/(num_runs * num_episodes))

# Plot learning
fig, ax = plt.subplots()
ax.plot(np.mean(acc_reward, axis=0))

ax.set(xlabel="Episode", ylabel="Mean accumulated reward",
       title="Q-Learning on Frozen Lake")
ax.grid()

fig.savefig("./learning_curves/learning_curve_1.png")

print("Plot of learning saved in: ./learning_curves/exercise1.png")