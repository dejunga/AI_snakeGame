import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()  # Initialize all imported pygame modules
font = pygame.font.Font('arial.ttf', 25)  # Set font for displaying score

# Enum to define directions for snake movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')  # A point represents the (x, y) coordinate on the grid

# Define colors using RGB values
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20  # Size of the snake block and food
SPEED = 4000000  # Speed of the game

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w  # Width of the game window
        self.h = h  # Height of the game window
        # Initialize the display window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')  # Set window title
        self.clock = pygame.time.Clock()  # Create a clock object to manage updates
        self.reset()  # Reset the game state

    def reset(self):
        # Initialize the game state
        self.direction = Direction.RIGHT  # Start direction
        self.head = Point(self.w / 2, self.h / 2)  # Start position of the snake's head
        # Initialize the snake body
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0  # Initialize score
        self.food = None  # Initialize food position
        self._place_food()  # Place the first food
        self.frame_iteration = 0  # Initialize frame iteration counter

    def _place_food(self):
        # Place food at a random location on the grid
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)  # Create a new Point for the food
        if self.food in self.snake:  # Ensure food does not overlap with snake
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1  # Increase frame iteration
        # 1. Handle user input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move the snake
        self._move(action)  # Update the snake's head position
        self.snake.insert(0, self.head)  # Add new head position to the snake body
        
        # 3. Check if game over
        reward = 0
        game_over = False
        # If collision occurs or the game runs for too many frames without eating
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # Negative reward for losing
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1  # Increase score
            reward = 10  # Positive reward for eating food
            self._place_food()  # Place new food
        else:
            self.snake.pop()  # Remove the last block of the snake

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)  # Control the speed of the game

        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Check if the snake hits the boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check if the snake hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)  # Fill the display with black color

        # Draw the snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Display the score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()  # Update the display

    def _move(self, action):
        # [straight, right, left]
        # Define clockwise directions
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)  # Find the index of the current direction

        # Determine the new direction based on the action
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left turn

        self.direction = new_dir  # Update direction

        # Update the head position based on the new direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)  # Update the head point

if __name__ == '__main__':
    game = SnakeGame()  # Create an instance of SnakeGame
    
    # Game loop
    while True:
        game_over, score = game.play_step()  # Play a step of the game
        
        if game_over == True:
            break  # Exit loop if game is over
        
    print('Final Score', score)  # Print the final score
        
    pygame.quit()  # Quit pygame
