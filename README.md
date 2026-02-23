# CS4100-Final-Project

Hanabi-playing Artificial Intelligence built using Deep Q-Learning.

Uses libraries: Pytorch, PettingZoo libraries.

Using Huggingface, Pytorch Documentation, as reference.

## Current Idea:
- Using a double DQN for target values (Target q values) and determining our next steps (Current q values, actual model).
- Replay memory to keep track of moves that occurred before; Need to figure out how we select our mask of moves to optimize with; 
- General process:
    1. Set up hyperparamters (Optimizer, loss function, epsilon for exploration/exploitation)
    2. Figure out action to take (epsilon exploration/exploitation)
    3. Take said action; optimize model as desc below
    4. Run the policy net to add transitions to memory and compute q-values.
    5. Choose a handful of transitions from memory (random), clean them up (separate into state, action, reward)
    6. Compute their target Q values for batch(using policy network)
    7. Compute the estimate Q values for batch using the (using target network)
    8. Compute the loss, then run backpropagation on the target network (for current q values)
    9. Using the value from the policy network as training data (computing loss).
    10. Update the policy network slightly (using hyperparameter TAU) to what target network is at.
    11. Repeat till cry. Then repeat again.

## Rough Deadlines:

- Starting working model ~3/13
  - Exploration/Exploitation Phase
  - Get a model working
  - Setup environment
  - 2-player
  - 5-color
  - 3-lives
  - 8-hints
- Optimization and Training ~3/31
- Slides 4/8
