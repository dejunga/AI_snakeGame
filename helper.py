import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # Turn on interactive mode in matplotlib

# Function to plot scores and mean scores during training
def plot(scores, mean_scores):
    display.clear_output(wait=True)  # Clear the previous plot
    display.display(plt.gcf())  # Display the current plot
    plt.clf()  # Clear the current figure
    plt.title('Training...')  # Set the title of the plot
    plt.xlabel('Number of Games')  # Set the x-axis label
    plt.ylabel('Score')  # Set the y-axis label
    plt.plot(scores)  # Plot the scores
    plt.plot(mean_scores)  # Plot the mean scores
    plt.ylim(ymin=0)  # Set the limit for y-axis
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))  # Annotate the last score
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))  # Annotate the last mean score
    plt.show(block=False)  # Show the plot without blocking
    plt.pause(.1)  # Pause to allow the plot to update
