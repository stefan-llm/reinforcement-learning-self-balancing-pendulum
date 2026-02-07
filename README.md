# I created a self balancing pendulum using reinforcment learning
I solved the inverted pendulum problem by using a Deep Q network implementing forward propogation and backward propogation, in my previous pendulum project 

# The learning process


# Formulas used in the project:
----
***Neural Network Class:***
  
He Weight Initialization:

$$W = \text{randn} \times \sqrt{\frac{2}{n_{in}}}$$
     
Forward Propagation:

$$z = A_{prev} \cdot W + B$$
$$A = \tanh(z) \quad \text{(hidden layers)}$$
$$A = z \quad \text{(output layer, linear)}$$
     
Backward Propagation / Gradient Descent:

$$\delta_{output} = A_{output} - target$$
$$\nabla W = \frac{A_{prev}^T \cdot \delta}{\text{batch size}}$$
$$\nabla B = \text{mean}(\delta)$$
$$\nabla W, \nabla B = \text{clip}(\nabla W, \nabla B, -1, 1)$$
$$W = W - lr \times \nabla W$$
$$B = B - lr \times \nabla B$$
$$\delta_{prev} = (\delta \cdot W^T) \times (1 - A_{prev}^2) \quad \text{(tanh derivative)}$$

---

***DQN Agent Class:***

Epsilon-greedy action:

$$a = \begin{cases} \text{random action} & \text{if } rand < \epsilon \ \arg\max(Q(s)) & \text{otherwise} \end{cases}$$

DQN Target:

$$Q_{target}(s, a) = \begin{cases} r & \text{if done} \ r + \gamma \cdot \max(Q{target\text{}net}(s')) & \text{otherwise} \end{cases}$$
$$\gamma = 0.99$$

Epsilon Decay:

$$\epsilon = \max(0.01, ; \epsilon \times 0.98)$$

---
***Physics (ENV) Class***

Angle normalisation:

$$\theta_{norm} = \text{atan2}(\sin\theta, \cos\theta)$$

State normalisation:

$$s = \left[\frac{x}{3}, ; \frac{\dot{x}}{5}, ; \theta_{norm}, ; \frac{\dot{\theta}}{5}\right]$$

Angular acceleration:

$$\ddot{\theta} = \frac{g \sin\theta \cdot (M + m) - \cos\theta \cdot (F + m L \dot{\theta}^2 \sin\theta)}{L \cdot (M + m\sin^2\theta)}$$

Cart acceleration:

$$\ddot{x} = \frac{F + mL(\dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta)}{M + m\sin^2\theta}$$

Euler Integration:

$$\dot{\theta} = \text{clip}(\dot{\theta} + \ddot{\theta} \cdot dt, ; -10, ; 10)$$
$$\theta = \theta + \dot{\theta} \cdot dt$$
$$\dot{x} = \text{clip}(\dot{x} + \ddot{x} \cdot dt, ; -10, ; 10)$$
$$x = x + \dot{x} \cdot dt$$

Reward function:

$$r = \begin{cases} -10 & \text{if } |\theta| > 0.4 \text{ or } |x| > 2.5 \ +1 & \end{cases} $$


---
***Visualisation***

Pendulum endpoint:

$$bx = cx + L \cdot scale \cdot \sin\theta$$
$$by = cy - L \cdot scale \cdot \cos\theta$$

Reward graph bar height:

$$h = \frac{reward}{\max(rewards)} \times 60$$












