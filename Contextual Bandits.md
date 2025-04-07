Notes from YT video: [Reinforcement Learning Chapter 2: Multi-Armed Bandits](https://www.youtube.com/watch?v=9LhNHK1ULxs)

---

Difference between RL and SL:
> RL evaluates actions taken rather than instructing with correct actions.


## k-armed bandits
![[Pasted image 20250401170823.png | 300]]
- Imagine 4 buttons and we get rewards by pressing each of the them. 
- Objective: explore and exploit so as to maximize the rewards over 100 pushes of the button
- In general, objective is to maximize the reward over a given numbers of time steps.

- Reward distribution
	- Stationary vs non-stationary rewards
		- stationary: reward does not changes wrt steps and time
	- Does the reward of pulling "lever-2" change over time?

- Maximizing reward
	- through repeated actions you maximize your winning by pulling the best levers?
	- how to find the best levers?
	- estimating values of each action 
		- one of the way is sample average method
$$
Q_t(a) \doteq \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}}
$$

- Greedy action selection rule
$$
A_t \doteq \arg\max\limits_{a} Q_t(a)
$$
- Greedy vs. $\epsilon$-greedy action selection
	- Discard $\arg\max_{a} Q_t(a)$ with probability $\epsilon$, and resample another action with uniform probability 
	- As the number of steps increases, every action will be sampled infinite number of times, ensuring that all the $Q_t(a)$ converges to $Q_*(a)$ 

- Efficient sample-averaging
	- constant memory and constant computation
	- if the average over 5 samples is 8, you only need to keep 5 and 40 to update the average in the future
$$
\begin{aligned}
Q_{n+1} &= \frac{1}{n} \sum_{i=1}^{n} R_i \\
        &= \frac{1}{n} \left( R_n + \sum_{i=1}^{n-1} R_i \right) \\
        &= \frac{1}{n} \left( R_n + (n-1) \frac{1}{n-1} \sum_{i=1}^{n-1} R_i \right) \\
        &= \frac{1}{n} \left( R_n + (n-1) Q_n \right) \\
        &= \frac{1}{n} \left( R_n + n Q_n - Q_n \right) \\
        &= Q_n + \frac{1}{n} \left[ R_n - Q_n \right]
\end{aligned}
$$
$$
{newEstimate} \leftarrow oldEstimate + StepSize\ [Target - OldEstimate]
$$

```ad-question
Q. Difference between Value function and Policy?
A. [[Value function Vs Reward function]]
```


- Greedy and $\epsilon$-greedy selection
	- Greedy
		- if reward variance = 0, greedy selection knows the value of each action after trying it once
	- $\epsilon$-greedy
		- with noiser rewards, it takes more exploration (eg. variance=10 vs 1)
		- non-stationary rewards