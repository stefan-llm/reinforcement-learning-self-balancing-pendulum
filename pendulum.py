import pygame, math, random, numpy as np, pickle, os
from collections import deque

# Initialise pygame screen width/height, synchronise clock
pygame.init()
W, H = 1000, 700
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Balancing pendulum using reinforcement learning")
screen.fill((0, 0, 0))
clock = pygame.time.Clock()

# Initialise colours
BLACK, WHITE, GRAY, RED, GREEN, BLUE, YELLOW, CYAN, ORANGE = (
    (20, 20, 30), (255, 255, 255), (100, 100, 100), (220, 80, 80), (80, 200, 120),
    (80, 140, 220), (240, 200, 80), (80, 200, 220), (240, 150, 50)
)

# the network can decide between 3 actions to move - left, right or stay still
FORCES = [-50, 0, 50]
N_ACTIONS = 3

# Neural Network class - the feedforward network built from scratch using NumPy
class NN:
    # Initialise weights using He Initialization and biases as zeros
    # sizes list defining neurons per layer
    def __init__(self, sizes):
        self.W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i]) for i in range(len(sizes)-1)]
        self.B = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]

    # Forward propagation - passes input through each layer
    # Applies tanh activation on hidden layers, linear on output layer
    def forward(self, x):
        self.A = [x]
        for i, (w, b) in enumerate(zip(self.W, self.B)):
            z = self.A[-1] @ w + b
            self.A.append(np.tanh(z) if i < len(self.W)-1 else z)
        return self.A[-1]

    # Backpropagation - computes error between output and target.
    # Propagates gradients backwards through each layer to update weights and biases.
    # Gradients are clipped to [-1, 1] preventing exploding gradients
    # Set the learning rate to 0.005
    def backward(self, target, lr=0.005):
        d = self.A[-1] - target
        for i in range(len(self.W)-1, -1, -1):
            gw = self.A[i].T @ d / len(target)
            gb = d.mean(0, keepdims=True)
            gw = np.clip(gw, -1, 1)
            gb = np.clip(gb, -1, 1)
            self.W[i] -= lr * gw
            self.B[i] -= lr * gb
            if i > 0:
                d = (d @ self.W[i].T) * (1 - self.A[i]**2)

    # Copies all weights and biases from another NN instance
    # Used to sync the target network with the main network
    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.B = [b.copy() for b in other.B]


# DQN Reinforcement_learning class the reinforcement learning agent that learns to balance the pendulum
class Reinforcement_learning:
    # Initialise the main network and a target network.
    # the target network only gets updated every 5 episodes as later it's used to calculate the Q targets in the bellman equasion.
    # initailise memory at 50000, epsilion, episode, rewards array, best, best test steps
    def __init__(self):
        self.net = NN([4, 24, 24, N_ACTIONS])  # small and fast
        self.target = NN([4, 24, 24, N_ACTIONS])
        self.target.copy_from(self.net)
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0
        self.episode = 0
        self.rewards = []
        self.best = 0
        self.best_test_steps = 0


    # Epsilon-greedy algorithm - picks a random action with probability
    # epsilon (exploration), otherwise picks the best action from the network (exploitation)
    def act(self, s, train=True):
        if train and random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS-1)
        return int(np.argmax(self.net.forward(np.array(s).reshape(1,-1))[0]))

    # Samples a batch of 64 experiences from replay buffer, computes DQN target
    # Q-values using the Bellman equation, then updates the network using backpropagation
    def train(self):
        if len(self.memory) < 64: return
        batch = random.sample(self.memory, 64)
        s, a, r, s2, d = map(np.array, zip(*batch))

        q = self.net.forward(s)
        q_next = self.target.forward(s2)

        target = q.copy()
        for i in range(len(batch)):
            target[i, a[i]] = r[i] if d[i] else r[i] + 0.99 * np.max(q_next[i])

        self.net.backward(target)

    # Automatically saves network weights, epsilon, episode count, rewards, and best scores to .pkl file
    def save(self):
        pickle.dump({'W': self.net.W, 'B': self.net.B, 'e': self.epsilon,
                     'ep': self.episode, 'r': self.rewards, 'b': self.best,
                     'bts': self.best_test_steps}, open('dqn.pkl', 'wb'))

    # Loads a previously saved model and restores all training state if the data exists
    def load(self):
        if os.path.exists('dqn.pkl'):
            try:
                d = pickle.load(open('dqn.pkl', 'rb'))
                self.net.W, self.net.B = d['W'], d['B']
                self.target.copy_from(self.net)
                self.epsilon, self.episode = d['e'], d['ep']
                self.rewards, self.best = d.get('r', []), d.get('b', -999)
                self.best_test_steps = d.get('bts', 0)
            except: pass


# Environment class - simulates the inverted pendulum on a cart
class Env:
    # Sets physics constants: gravity, pendulum length, pendulum mass, cart mass
    def __init__(self):
        self.g, self.L, self.m, self.M = 9.81, 2.0, 0.5, 2.0
        self.scale, self.track_y = 80, H - 150
        self.reset()

    # Resets the environment - cart to centre, angle randomised between -0.3 and 0.3 radians
    def reset(self):
        self.x = self.x_dot = self.theta_dot = 0
        self.theta = random.uniform(-0.3, 0.3)
        self.steps = self.total_r = 0
        return self.state()

    # Returns the current observation as a normalised 4-element array
    # Angle is wrapped to [-pi, pi] using atan2
    def state(self):
        th = math.atan2(math.sin(self.theta), math.cos(self.theta))
        return np.array([self.x/3, self.x_dot/5, th, self.theta_dot/5], dtype=np.float32)

    # Applies a force to the cart and advances physics by 4 sub-steps (0.004s each)
    # Returns (next_state, reward, done). Reward: +1 alive, -10 if fallen
    # Training ends after 500 steps or failure; testing only ends on failure
    def step(self, action, training=True):
        force = FORCES[action]

        for _ in range(4):
            s, c = math.sin(self.theta), math.cos(self.theta)
            den = self.M + self.m*s*s + 0.0001
            th_dd = (self.g*s*(self.M+self.m) - c*(force + self.m*self.L*self.theta_dot**2*s)) / (self.L*den)
            x_dd = (force + self.m*self.L*(self.theta_dot**2*s - th_dd*c)) / den
            self.theta_dot = np.clip(self.theta_dot + th_dd*0.004, -10, 10)
            self.theta += self.theta_dot * 0.004
            self.x_dot = np.clip(self.x_dot + x_dd*0.004, -10, 10)
            self.x += self.x_dot * 0.004

        self.x = np.clip(self.x, -3, 3)
        self.steps += 1
        th = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # REWARD: +1 alive, penalise cart drift and velocity to minimise movement
        r = 1.0 - 0.3 * (self.x / 2.5)**2 - 0.2 * (self.x_dot / 5)**2

        # tight angle = learns to stay very upright
        fallen = abs(th) > 0.4 or abs(self.x) > 2.5

        if training:
            done = fallen or self.steps >= 500
        else:
            done = fallen

        # Given a big penalty of -10 compared to the reward of 1
        if fallen:
            r = -10

        self.total_r += r
        return self.state(), r, done

    # Renders the pendulum, cart, track, and force arrow on screen
    # Pendulum colour: green (< 0.1 rad), yellow (< 0.3 rad), red (> 0.3 rad)
    def draw(self, action=1):
        cx, cy = W//2 + int(self.x*self.scale), self.track_y
        bx = cx + int(self.L*self.scale*math.sin(self.theta))
        by = cy - int(self.L*self.scale*math.cos(self.theta))
        pygame.draw.line(screen, GRAY, (100, self.track_y), (W-100, self.track_y), 4)
        if FORCES[action] != 0:
            pygame.draw.line(screen, CYAN, (cx, cy+30), (cx + FORCES[action], cy+30), 3)
        pygame.draw.rect(screen, BLUE, (cx-30, cy-12, 60, 24), border_radius=5)
        pygame.draw.line(screen, WHITE, (cx, cy), (bx, by), 3)
        th = abs(math.atan2(math.sin(self.theta), math.cos(self.theta)))
        pygame.draw.circle(screen, GREEN if th < 0.1 else YELLOW if th < 0.3 else RED, (bx, by), 12)


# Main game loop - handles keyboard input, runs the agent's action-train loop,
# displays stats on screen, and renders a reward history graph
def main():
    env, agent = Env(), Reinforcement_learning()
    agent.load()

    font = pygame.font.SysFont('Arial', 18)
    state, action = env.reset(), 1
    training = True
    paused = False
    fast = False

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: paused = not paused
                if e.key == pygame.K_t: training = not training; state = env.reset()
                if e.key == pygame.K_f: fast = not fast

        if not paused:
            for _ in range(50 if fast else 1): # 50 times speed in fast mode
                action = agent.act(state, training)
                next_state, reward, done = env.step(action, training)
                if training:
                    agent.memory.append((state, action, reward, next_state, done))
                    agent.train()
                state = next_state
                if done:
                    agent.rewards.append(env.total_r)
                    agent.best = max(agent.best, env.total_r)
                    if not training:
                        agent.best_test_steps = max(agent.best_test_steps, env.steps)
                    if training:
                        agent.episode += 1
                        # Fast epsilon decay
                        agent.epsilon = max(0.01, agent.epsilon * 0.98)
                        if agent.episode % 5 == 0:
                            agent.target.copy_from(agent.net)
                        if agent.episode % 50 == 0:
                            agent.save()
                    state = env.reset()
                    break

        # Always render even in fast mode
        if not fast or agent.episode % 2 == 0:
            screen.fill(BLACK)
            mode = "TRAINING" if training else "TESTING (infinite)"
            screen.blit(font.render(f"DQN Pendulum - {mode}", True, ORANGE if training else GREEN), (W//2-100, 10))
            env.draw(action)

            avg = np.mean(agent.rewards[-20:]) if agent.rewards else 0
            stats = [
                f"Episode: {agent.episode}",
                f"Epsilon: {agent.epsilon:.0%}",
                f"Steps: {env.steps}",
                f"Reward: {env.total_r:.0f}",
                f"Avg(20): {avg:.0f}",
                f"Highest reward: {agent.best:.0f}",
                f"Highest balance time(during training): {agent.best_test_steps * 0.016:.1f}s at {agent.best_test_steps} steps"
            ]

            for i, s in enumerate(stats):
                screen.blit(font.render(s, True, WHITE), (20, 20 + i*25))

            screen.blit(font.render("SPACE - pause  T - test  F - fast", True, GRAY), (W-400, H-30))

            # Simple bar graph
            if len(agent.rewards) > 10:
                pts = agent.rewards[-150:]
                mx = max(pts) if pts else 1
                for i, p in enumerate(pts):
                    h = int(p / max(mx, 1) * 60)
                    color = GREEN if p > 400 else YELLOW if p > 200 else RED
                    pygame.draw.line(screen, color, (20+i, 300), (20+i, 300-h), 2)

            pygame.display.flip()
        clock.tick(60 if not fast else 0)

    agent.save()
    pygame.quit()

main()
