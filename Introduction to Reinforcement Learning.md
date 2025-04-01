Notes from YT video: [An Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0)

![[Pasted image 20250401124804.png | 500]]

- One of the most promising subfields in ML is RL to get to very intelligent robotic behaviour
- The most common type of ML is Supervised learning
	- we give an input to NN model along with expected output label
	- compute gradients using back propagation algorithm to train NN
- Imagine training NN to play game of pong in a supervised setting
	- have a good human gamer play the game of pong for couple for hours
	- create a dataset to log all the frames that human is seeing on screen as well as the action that he takes in response to those frames (↑ or ↓)
	- feed these frames to NN that in the output produces 2 simple actions i.e. selection of up or down arrows
	- this replicates the action of human gamer. But, it 2 major ==downsides== to this approach:
		- creation of the dataset. Might not be easy or feasible all the time
		- if you train you NN model to replicate the actions of human player then by definition your agent can _never_ be better than the player
![[Pasted image 20250401130444.png | 400]]


- Is their a way for agent to learn the play a game entirely by itself?
	- well, this is called RL!
- The framework of RL is surprisingly similar to normal framework in Supervised learning.
	- we still have an input frame and we run it through NN and model produces an output action (↑ or ↓) but the only difference is that now we do not actually know the target label (no idea of whether we should have gone ↑ or ↓). 
- In RL, the network that transforms input frames to output actions is called the "*Policy Network*".
- One of simplest way to train a Policy Network is a method called _Policy Gradients_.
	- Approach in PG is that you start out with a completely random network. 
	- You feed that network a frame from the game engine, it produces a random action (either ↑ or ↓). 
	- You send the action back to the game engine and the game engine produces the next frame. 
	- And the loop continues.
	- Output contain 2 numbers, probability of going up (↑) and of going down (↓)
	- What you will do while training is to sample from the distribution so that you're not always going to repeat the same exact actions and this will allow your agents to sort of explore the environment a bit randomly and hopefully discover better rewards and behaviour. 
	- Since we want the agent to learn on its own the only feedback that we pass is the game scoreboard. 
		- if agent manages to score a goal it receives a reward of $+1$.
		- if opponent scores a goal then our agent receives a penalty of $-1$.
		- goal of agent is to maximize its policy to receive as much as possible. 



![[Pasted image 20250401143217.png | 400]]
- In order to train the PN, 1st thing we need to do is to collect a bunch of experience so we're just going to run a whole bunch of those game frames through the network, select random action, feed them back into the engine, and create a whole bunch of random pong games.
	- Most of the time the agent will lose the game as it has learned nothing but sometimes it wins by randomly selecting a sequence of action that lead him to score a goal. In this case, it receives a reward.
	![[Pasted image 20250401150516.png | 450]]

- Key thing to understand is that, for every episode; regardless of whether we want a $+ve$ or $-ve$ rewards we can compute gradients that would make the actions that our agent has chosen more likely in the future. And this is very crucial. 
	- What policy gradients are going to do is that for every episode where we've got a $+ve$ reward we're going to use the normal gradients to increase the probability of those actions in future. 
	- Whenever we get a $-ve$ reward we're going to apply the same gradient but we'll multiply it with $-1$. This $(-)$ sign will make sure that in the future all the actions that we took in a bad episode are going to be less likely in the future.
![[Pasted image 20250401145149.png]]
- The result is that while training our PN the actions that lead to $-ve$ rewards are slowly going to be filtered out and the actions that lead of $+ve$ rewards are going to become more and more likely. 
	- In this sense our agent learns to play the game of pong. 

#### Significant downsides of using Policy Gradient
- Suppose in a training session, if agent loses the episode it considers that all the actions performed lead to the loss and apparently it reduces the likelihood of taking those actions in the future.
	- In RL scenario this is called as _Credit Assignment Problem_. 
	- This happens because we give reward only towards the end of episode and not for each action (_Sparse Reward Setting_) and the agent needs to figure what action sequence are causing the reward it eventually gets. 

### Reward shaping
- The traditional approach to solve the issue of Sparse rewards has been the use of _Reward Shaping_.
- It is the process of manually designing a reward function that need to guide your policy to some desired behaviour. 
- This obviously help the policy to converge to desired behaviour, there are still significant downsides to it:
	- hard to design the reward function
	- need to custom design the policy for every new environment where you want to train the policy. Thus not scalable.
	- suffers from _Alignment Problem_. It turns out that in several cases agent learns to earn lot of rewards without actually accomplishing desired tasks (bypasses the actual goal). 
		- policy is just overfitting to that specific reward function that you designed while not generalising to the intended behaviour. 


### Thoughts
- Training with sparse rewards → really hard!
- Reward shaping → not an optimal solution


### Reference articles
- [Deep Reinforcement Learning: Pong from Pixels](https://karpathy.github.io/2016/05/31/rl/)
	- Code gist: [Policy Gradients- Pong](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)





---
## Legends

| Short hand | Full form              |
| ---------- | ---------------------- |
| RL         | Reinforcement Learning |
| ML         | Machine Learning       |
| NN         | Neural Network         |
| PG         | Policy Gradients       |
| PN         | Policy Network         |


