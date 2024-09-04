import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Define a neural network model with one hidden layer for Q-learning
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()  # Initialize the base class
        self.linear1 = nn.Linear(input_size, hidden_size)  # Define the first linear layer
        self.linear2 = nn.Linear(hidden_size, output_size)  # Define the second linear layer

    def forward(self, x):
        # Define the forward pass of the network
        x = F.relu(self.linear1(x))  # Apply ReLU activation to the output of the first layer
        x = self.linear2(x)  # Pass through the second layer
        return x

    def save(self, file_name='model.pth'):
        # Save the model parameters to a file
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# Trainer class for training the Q-network
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor for future rewards
        self.model = model  # The Q-network to be trained
        self.target_model = Linear_QNet(model.linear1.in_features, model.linear1.out_features, model.linear2.out_features)
        self.update_target_model()  # Initialize target model with same weights as model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Adam optimizer
        self.criterion = nn.MSELoss()  # Mean Squared Error loss function

    def update_target_model(self):
        # Update the target model's weights with the current model's weights
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        # Convert data to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # If the state is a single example, add a batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q-values with the current state
        pred = self.model(state)

        target = pred.clone()  # Clone the predicted Q-values
        action_indices = [torch.argmax(action[idx]).item() for idx in range(len(done))]
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Calculate the new Q-value using the Bellman equation
                Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx])).detach()
            
            target[idx][action_indices[idx]] = Q_new  # Update the target Q-value

        # Backpropagate the loss and update the model weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
