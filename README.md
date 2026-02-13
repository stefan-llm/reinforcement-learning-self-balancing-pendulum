# Inverted Pendulum — Deep Q-Network from Scratch

I solved the inverted pendulum problem using reinforcement learning, implementing a Deep Q-Network with forward propagation and backward propagation entirely from scratch — no ML frameworks.

## The Problem

The inverted pendulum is a classic control problem where an agent must balance a pole upright on a cart by applying left or right forces. The system is inherently unstable — without continuous correction, the pole falls.

## State & Action Space

The environment provides a state consisting of four observations: cart position, cart velocity, pole angle, and pole angular velocity. At each timestep, the agent chooses a discrete action — push left or push right.

## Forward Propagation

Passes the current state through the network's layers to produce Q-value estimates for each possible action. The agent selects the action with the highest Q-value (with epsilon-greedy exploration to balance exploitation and exploration).

## Backward Propagation

Updates the network weights by minimising the loss between predicted Q-values and target Q-values derived from the Bellman equation:

`Q_target = reward + γ * max(Q_target_net(s', a'))`

where γ is the discount factor, s' is the next state, and the target network provides stable Q-value estimates.

I've found that in my case, somewhere between episodes 500 to 1000 is where the pendulum can balance perfectly when switched to test mode.

---

## Controls

| Key | Action |
|-----|--------|
| **Space** | Pause |
| **T** | Switch between train and test mode |
| **F** | Toggle 50x speed for faster training |

## Usage

Train the agent until the reward graph turns green like this:

<img width="354" height="220" alt="Reward graph showing successful training" src="https://github.com/user-attachments/assets/41cc1804-7d54-42d0-b5af-31186f62d8c9" />

Then press **F** to stop fast-forward, and press **T** to switch to test mode and watch the pendulum balance indefinitely.

---

## How It Learns

**Initial State:** The network starts with completely random weights, so it has no idea what to do. It's like a baby learning to walk — it makes random guesses and learns from the results.

### 1. Exploration vs Exploitation

The agent uses an epsilon-greedy strategy. At first (ε = 100%), it explores by taking completely random actions. As training progresses, epsilon decays to 1%, meaning it increasingly exploits what it has learned. This balance ensures it discovers good strategies early on while eventually settling into optimal behaviour.

### 2. Experience Replay

Every action the agent takes is stored in a memory buffer (max 50,000 experiences). During training, it randomly samples batches of 64 past experiences. This breaks correlations between consecutive states and makes learning more stable — it's like studying from shuffled flashcards instead of reading the same chapter repeatedly.

### 3. The Reward Signal

The agent receives +1 for every timestep it stays balanced and -10 when it falls. That's the only supervision it gets. Using the Bellman equation, it learns to propagate these rewards backwards through time, discovering that states leading to falls are bad and states maintaining balance are good.

### 4. Credit Assignment

When the agent falls, the DQN algorithm traces back through recent experiences and updates Q-values for states that led to the failure. States further from the failure get smaller updates (scaled by γ = 0.99). This teaches the network to predict not just immediate rewards, but cumulative future rewards.

### 5. Target Network

The agent maintains two networks — a main network that's constantly updated, and a target network that's only synced every 5 episodes. This stabilises training by preventing the "moving target" problem where the network chases its own changing predictions.

### 6. Improvement Over Time

For an average run:

- **Episodes 1–50:** Random flailing, falls quickly
- **Episodes 50–200:** Discovers that keeping θ ≈ 0 is good
- **Episodes 200–500:** Learns smooth corrective actions
- **Episodes 500+:** Masters the task, can balance indefinitely

---

## Reward Shaping Experiment

During development, I experimented with minimising how often the network pushes left and right, and maximising how long the pendulum stays upright with minimal corrections. I did this by replacing the simple +1 reward with a shaped reward:

`r = 1.0 - 0.3 * (x / 2.5)² - 0.2 * (x_dot / 5)²`

This penalises the cart for drifting from centre and for moving fast, encouraging the agent to learn smoother, more minimal corrections rather than chaotic balancing.

---

## Formulas

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
