# I created a self balancing pendulum using reinforcment learning
I solved the inverted pendulum problem by using a Deep Q network implementing forward propogation and backward propogation.

## The Problem
  The inverted pendulum is a classic control problem where an agent
  must balance a pole upright on a cart by applying left or right
  forces. The system is inherently unstable — without continuous
  correction, the pole falls.
  
## State & Action Space
  The environment provides a state consisting of four observations:
  cart position, cart velocity, pole angle, and pole angular
  velocity. At each timestep, the agent chooses a discrete action —
  push left or push right.

## Forward propagation
  passes the current state through the
  network's layers to produce Q-value estimates for each possible
  action. The agent selects the action with the highest Q-value
  (with epsilon-greedy exploration to balance exploitation and
  exploration).

## Backward propagation
  updates the network weights by minimising
  the loss between predicted Q-values and target Q-values derived
  from the Bellman equation:
  Q_target = reward + γ * max(Q(s', a')), where γ is the discount
  factor and s' is the next state.

Ive found that in my case, steps between 500 to 1000 is where the pendulum can balance perfectly when switched to test mode

# How it works
    Space key - Pause
    
    T key - switch from train to test / test to train
    
    F key - Speed up the learning process

  As soon as the graph goes to green such as this -  <img width="354" height="220" alt="image" src="https://github.com/user-attachments/assets/41cc1804-7d54-42d0-b5af-31186f62d8c9" />

  
  Then press the F to stop 50x speed, then press the T key to test the pendulum and watch it balance indefinitely.


***Initial State:***

  The network starts with completely random weights, so it has no idea what to do. It's like a baby learning to walk - it makes random guesses and learns from the results.

1. Exploration vs exploitation

The agent uses an epsilon-greedy strategy. At first (ε = 100%), it explores by taking completely random actions. As training progresses, epsilon decays to 1%, meaning it increasingly exploits what it has learned. This balance ensures it discovers good strategies early on while eventually settling into optimal behavior.

2. Experience replay

Every action the agent takes is stored in a memory buffer (max 50,000 experiences). During training, it randomly samples batches of 64 past experiences. This breaks correlations between consecutive states and makes learning more stable - it's like studying from shuffled flashcards instead of reading the same chapter repeatedly.

3. The reward signal

The agent receives +1 for every timestep it stays balanced and -10 when it falls. That's the only supervision it gets. Using the Bellman equation, it learns to propagate these rewards backwards through time, discovering that falling is bad and standing up is good.

4. Credit assignment

When the agent falls, the DQN algorithm traces back through recent experiences and updates Q-values for states that led to the failure. States further from the failure get smaller updates (scaled by γ = 0.99). This teaches the network to predict not just immediate rewards, but cumulative future rewards.

5. Target network

The agent maintains two networks - a main network that's constantly updated, and a target network that's only synced every 5 episodes. This stabilizes training by preventing the "moving target" problem where the network chases its own changing predictions.

6. Improvements over time

****For an average run****

    Episodes 1-50: Random flailing, falls quickly

    Episodes 50-200: Discovers that keeping θ ≈ 0 is good
    
    Episodes 200-500: Learns smooth corrective actions
    
    Episode 500+: Masters the task, can balance indefinitely


## Important ideas
During the development process i had an idea of minimising the amount of times the network pushes left and right and maximising the amount of time the pendulum is up the way i did this was through the way i rewarded the network - r = 1.0 - 0.3 * (self.x / 2.5)**2 - 0.2 * (self.x_dot / 5)**2 - This replaced the original r = 1.0. The idea was to penalise the cart for drifting from centre and for moving fast, so the agent learns smoother, more minimal corrections. The reason i did this was to solve the problem of chaotic balancing.

# Formulas used in the project:
----
### Neural Network

**He Weight Initialisation:**

$$W = \text{randn} \times \sqrt{\frac{2}{n_{in}}}$$

**Forward Propagation:**

$$z = A_{prev} \cdot W + B$$
$$A = \tanh(z) \quad \text{(hidden layers)}$$
$$A = z \quad \text{(output layer, linear)}$$

**Backward Propagation / Gradient Descent:**

$$\delta_{output} = A_{output} - target$$
$$\nabla W = \frac{A_{prev}^T \cdot \delta}{\text{batch size}}$$
$$\nabla B = \text{mean}(\delta)$$
$$\nabla W, \nabla B = \text{clip}(\nabla W, \nabla B, -1, 1)$$
$$W = W - lr \times \nabla W$$
$$B = B - lr \times \nabla B$$
$$\delta_{prev} = (\delta \cdot W^T) \times (1 - A_{prev}^2) \quad \text{(tanh derivative)}$$

### DQN Agent

**Epsilon-Greedy Action Selection:**

$$a = \begin{cases} \text{random action} & \text{if } rand < \epsilon \\ \arg\max(Q(s)) & \text{otherwise} \end{cases}$$

**DQN Target (Bellman Equation):**

$$Q_{target}(s, a) = \begin{cases} r & \text{if done} \\ r + \gamma \cdot \max(Q_{target\_net}(s')) & \text{otherwise} \end{cases}$$
$$\gamma = 0.99$$

**Epsilon Decay:**

$$\epsilon = \max(0.01, \; \epsilon \times 0.98)$$

### Physics

**Angle Normalisation:**

$$\theta_{norm} = \text{atan2}(\sin\theta, \cos\theta)$$

**State Normalisation:**

$$s = \left[\frac{x}{3}, \; \frac{\dot{x}}{5}, \; \theta_{norm}, \; \frac{\dot{\theta}}{5}\right]$$

**Angular Acceleration:**

$$\ddot{\theta} = \frac{g \sin\theta \cdot (M + m) - \cos\theta \cdot (F + m L \dot{\theta}^2 \sin\theta)}{L \cdot (M + m\sin^2\theta)}$$

**Cart Acceleration:**

$$\ddot{x} = \frac{F + mL(\dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta)}{M + m\sin^2\theta}$$

**Euler Integration:**

$$\dot{\theta} = \text{clip}(\dot{\theta} + \ddot{\theta} \cdot dt, \; -10, \; 10)$$
$$\theta = \theta + \dot{\theta} \cdot dt$$
$$\dot{x} = \text{clip}(\dot{x} + \ddot{x} \cdot dt, \; -10, \; 10)$$
$$x = x + \dot{x} \cdot dt$$

**Reward Function:**

$$r = \begin{cases} -10 & \text{if } |\theta| > 0.4 \text{ or } |x| > 2.5 \\ +1 & \text{otherwise} \end{cases}$$

### Visualisation

**Pendulum Endpoint:**

$$bx = cx + L \cdot scale \cdot \sin\theta$$
$$by = cy - L \cdot scale \cdot \cos\theta$$

**Reward Graph Bar Height:**

$$h = \frac{reward}{\max(rewards)} \times 60$$


