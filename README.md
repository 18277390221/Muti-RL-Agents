# Improving PPO for Multi‑Agent Soccer in Unity ML‑Agents 3v3 Environment

Multi-agent reinforcement learning (MARL) has seen rapid progress in recent years, empowering teams of AI agents to master complex collaborative and competitive tasks. A prominent testbed is the Unity ML-Agents 3v3 Soccer environment. Proximal Policy Optimization (PPO) is a widely used in single-agent RL. However, applying PPO to multi-agent scenarios like 3v3 soccer raises issues of non-stationarity that are not present in single-agent settings. This project aim to survey extensions to PPO tailored for multi-agent learning and hope to find innovative enhancements to improve PPO in this environment.

## Background

### Environment Setup

The [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) 3v3 Soccer environment is a physics-based multi-agent simulation in which two teams of three agents each compete in a soccer match. 

Each soccer agent has a **partial observation** of the field. Instead of a global view, agents rely on egocentric sensor data.

Agents receive a $+1$ reward for scoring a goal against the opponent, and a $-1$ reward if a goal is scored against their own team. The reward is typically shared among teammates, which aligns the teammates' incentives completely and defines a zero-sum competitive game between the two teams. The environment is thus a mix of **competitive and cooperative dynamics** where agents cooperate to achieve the common goal within each team, while two teams are adversary. 

Unity 3v3 Soccer environment provides a controlled yet challenges domain for MARL research.

### Proximal Policy Optimization (PPO)

[Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) is a policy-gradient reinforcement learning algorithm introduced by Schulman et al. as an efficient, stable improvements on [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/2109.11251).

Let $r_t(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$. TRPO maximizes a "surrogate" objective:

$L^{CPI}(\theta)=\hat{E}_t\big[r_t(\theta)\hat{A}_t\big]$

Without a constraint, maximization of $L^{CPI}$ would lead to an excessively large policy update. PPO modify the objective to penalize changes to the policy that move $r_t(\theta)$ away from 1:

$L^{CLIP}(\theta)=\hat{E}_t\bigg[\min(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\bigg]$

Intuitively, PPO's clipping acts as a soft trust region: it prevents big policy jumps could collapse performance, while it allow more flexibility than hard constraints like in TRPO.

## Multi-Agent Challenges

Applying PPO to the 3v3 soccer environment yields insights into both the algorithm’s robustness and the difficulties posed by multi-agent learning. 

### Stability and Non-Stationarity

In multi-agent learning, each agent’s policy is part of other agents’ observed environment. As agents update their policies, the environment’s dynamics as seen by other agent are non-stationary. PPO’s stable update alone does not solve this. Agents can end up chasing each other’s changing strategies, and the training may oscillate or diverge without special care.

### Coordination and Teamwork

One of the key performance aspects in 3v3 soccer is whether agents learn to coordinate effectively. Vanilla PPO does not have an intrinsic mechanism for inducing coordination.  In a fully cooperative setting, independent PPO agents could learn to cooperate if the reward signals properly credit joint behavior. However, credit assignment is a challenge.  PPO’s advantage estimation might not correctly attribute which actions or which agent’s behavior led to the goal. 

### Sample Efficiency

Off-policy algorithms like DQN-based or MADDPG often claim better sample efficiency than PPO by reusing past experience. However, [recent benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf) has shown that PPO can be quite competitive in sample efficiency for MARL when implemented with large batch sizes and proper tunning.

## Basic extensions of PPOs for MARL

### Self-Play

Self-play pits agents against past version of themselves as opponents, ensuring that as one team improves, difficult of the opposing also increases. Some of the most notable successes in MARL, like OpenAI Five (Dota2) and DeepMind's AlphaStar (StarCraft II), relied on massive-scale self-play and league training. 

### Centralized Training, Decentralized Execution (CTDE)

One widely used idea in MARL is centralized training with decentralized execution (CTDE). [Multi-Agent PPO (MAPPO)](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf) extend PPO to this paradigm by introduce a centralized critic. Empirically, this design has been shown to achieve strong performance in a wide variety of cooperative multi-agent settings.

### Reward Shaping

Reward shaping involves adding additional reward signals to guide the agents toward desired behaviors, without changing the optimal policy of the underlying game. In soccer, beyond the primary rewards for goals, we might give small rewards for intermediate objectives like maintaining ball possession, making successful passes, moving the ball toward the opponent’s goal... Reward shaping helps overcome sparse rewards and can initiate complex behaviors.

#### [Update] Designing a Dense Reward Function with Language Model
The original Soccer environment used very sparse rewards: $+1$ for scoring a goal while $-1$ if opponents score a goal. Such sparse signals make learning slow in complex environments. By designing a dense reward, we provide incremental feedback for subgoals (e.g. gaining possession, passing, defending) that guide agents toward the final objective. However, doing this manually via trial-and-error is difficult and time consuming and often finding them sub-optimal. This is where a powerful language model can help.


PPO is a viable baseline for multi-agent training. It brings stable improvements and, with self-play, can solve the task of learning to play soccer at a reasonable skill level. However, certain aspects like efficient teamwork, faster convergence, and strategical depth may not fully emerge with naive PPO alone. There is room to incorporate domain-specific modifications and recent MARL innovations to push performance further.

## Potential extensions of PPOs for MARL

### Opponent Modeling

One useful extension for competitive multi-agent is opponent modeling, where agents explicitly build a model or estimate of other agents’ strategies and use it in their decision-making. There are MARL algorithms specifically designed for opponent modeling, such as [Learning with Opponent-Learning Awareness (LOLΑ)](https://arxiv.org/abs/1709.04326) and various meta-learning approaches.

In self-play PPO, the policy implicitly adapts to the opponent through the learning process, but it doesn’t maintain a separate mental model of the opponent’s policy. Opponent modeling techniques would allow an agent to, for instance, predict what an opposing player will do and then choose an action best responding to that prediction. This can be helpful in non-stationary settings or when facing unknown opponents.

While these are not part of standard PPO, they could be combined. [OPS-DeMo](https://arxiv.org/pdf/2406.06500) introduce an online algorithm and utilize running error estimation metric to detect policy switch of opponent, enabling the effective use of algorithms like PPO in multi-agent environment. 

A potential candidate is to modify the PPO loss to include terms that come from an opponent model, like anticipating opponent’s gradient steps.

### Population-based Training

In standard self-play, we have one current policy learning against a past version or a mirror of itself. [Population-Based Training (PBT)](https://arxiv.org/abs/1711.09846) would maintain a population of diverse policies for each team, periodically selecting the most successful ones, mutating hyperparameters, and mixing strategies.

A potential candidate is to enhance self-play with PBT and league dynamics.

## Group members

* Group Leader: Ziyuan Jin 2024233160

* Group members: Hanjia Cui 2024233130, Bowei Li 2024232099