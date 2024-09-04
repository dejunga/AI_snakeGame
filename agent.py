import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point  # Import game-related classes and utilities
from model import Linear_QNet, QTrainer  # Import neural network model and trainer
from helper import plot  # Import plot function for visualizing training progress
import pandas as pd  # Import pandas for data manipulation and saving

MAX_MEMORY = 100_000  # The maximum number of experiences to store in memory for experience replay.
BATCH_SIZE = 1000  # The number of experiences sampled from memory for each training step.
LR = 0.001  # Learning rate for the optimizer.

class Agent:

    def __init__(self):
        self.n_games = 0  # Counter for the number of games played.
        self.epsilon = 0  # Controls randomness in the agent’s actions (exploration vs. exploitation).
        self.gamma = 0.9  # Discount factor for future rewards, balancing short-term and long-term gains.
        self.memory = deque(maxlen=MAX_MEMORY)  # Stores past experiences for replay, improving training efficiency by breaking correlation between consecutive actions.
        self.model = Linear_QNet(11, 256, 3)  # A neural network defined in model.py for predicting Q-values.
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # A helper class defined in model.py for training the model.
        self.data = []  # List to store data for exploratory data analysis (EDA).

    def get_state(self, game):
        head = game.snake[0]  # Get the current position of the snake's head
        # Calculate points around the head to detect possible collisions
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Boolean flags for current direction of the snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # Is food to the left?
            game.food.x > game.head.x,  # Is food to the right?
            game.food.y < game.head.y,  # Is food above?
            game.food.y > game.head.y   # Is food below?
        ]

        return np.array(state, dtype=int)  # Return the state as a numpy array for the model to process

    def remember(self, state, action, reward, next_state, done):
        # Save a tuple of experience (state, action, reward, next_state, done) to memory for later training
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Train the model on a batch of experiences sampled from memory to improve generalization
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        # Unpack the mini-sample into separate lists for each element
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # Train the model using the mini-sample
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model on the most recent experience to allow for immediate learning
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Decide the next action using an epsilon-greedy strategy
        self.epsilon = 80 - self.n_games  # Decaying epsilon for less exploration over time
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # Exploration: Choose a random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: Choose the action with the highest predicted Q-value
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def save_data(self):  # Function to save data
        # Save collected data into a CSV file for further analysis
        df = pd.DataFrame(self.data, columns=['state', 'action', 'reward', 'next_state', 'done'])
        df.to_csv('game_data.csv', index=False)

def train():
    plot_scores = []  # List to store scores for plotting
    plot_mean_scores = []  # List to store mean scores for plotting
    total_score = 0  # Initialize total score
    record = 0  # Initialize record score
    agent = Agent()  # Create an instance of the Agent class
    game = SnakeGameAI()  # Create an instance of the SnakeGameAI class
    while True:
        # Get the current state of the game
        state_old = agent.get_state(game)

        # Decide the next action
        final_move = agent.get_action(state_old)

        # Perform the action and get the new state and reward
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train the model with the short-term memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember the experience for long-term memory
        agent.remember(state_old, final_move, reward, state_new, done)
        
        # Collect data for exploratory data analysis (EDA)
        agent.data.append((state_old, final_move, reward, state_new, done))

        if done:
            # If the game is over, reset the environment and update the agent
            game.reset()
            agent.n_games += 1  # Increment the number of games played
            agent.train_long_memory()  # Train the model with the long-term memory

            if score > record:
                record = score  # Update the record if the score is higher
                agent.model.save()  # Save the model with the new record

            print('Game', agent.n_games, 'Score', score, 'Record:', record)  # Print the game statistics

            # Update the target model every 10 games
            if agent.n_games % 10 == 0:
                agent.trainer.update_target_model()

            plot_scores.append(score)  # Add the score to the plot list
            total_score += score  # Update the total score
            mean_score = total_score / agent.n_games  # Calculate the mean score
            plot_mean_scores.append(mean_score)  # Add the mean score to the plot list
            plot(plot_scores, plot_mean_scores)  # Plot the scores
            
            # Save data after every 10 games
            if agent.n_games % 10 == 0:
                agent.save_data()

if __name__ == '__main__':
    train()  # Start training the agent
