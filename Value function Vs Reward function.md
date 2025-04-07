
- [[#Difference b/w Reward and Value function|Difference b/w Reward and Value function]]
	- [[#Difference b/w Reward and Value function#➳ Reward Function — "What do I get _now_?"|➳ Reward Function — "What do I get _now_?"]]
		- [[#➳ Reward Function — "What do I get _now_?"#> Definition|> Definition]]
		- [[#➳ Reward Function — "What do I get _now_?"#> Purpose|> Purpose]]
		- [[#➳ Reward Function — "What do I get _now_?"#> Example|> Example]]
	- [[#Difference b/w Reward and Value function#➳ Value Function — "How good is this situation _in the long run_?"|➳ Value Function — "How good is this situation _in the long run_?"]]
		- [[#➳ Value Function — "How good is this situation _in the long run_?"#> Definition|> Definition]]
		- [[#➳ Value Function — "How good is this situation _in the long run_?"#> Purpose|> Purpose]]
		- [[#➳ Value Function — "How good is this situation _in the long run_?"#> Example|> Example]]
	- [[#Difference b/w Reward and Value function#Analogy: Playing Chess|Analogy: Playing Chess]]
- [[#More about "Rewards"|More about "Rewards"]]
	- [[#More about "Rewards"#What does "immediate" rewards mean?|What does "immediate" rewards mean?]]
	- [[#More about "Rewards"#Chess example - clarified|Chess example - clarified]]
- [[#Difference b/w Reward & Value function: Chess example|Difference b/w Reward & Value function: Chess example]]
	- [[#Difference b/w Reward & Value function: Chess example#The Reward function in this case|The Reward function in this case]]
		- [[#The Reward function in this case#➳ Definition|➳ Definition]]
		- [[#The Reward function in this case#➳ Key characteristics|➳ Key characteristics]]
	- [[#Difference b/w Reward & Value function: Chess example#The Value function in this case|The Value function in this case]]
		- [[#The Value function in this case#➳ Definition|➳ Definition]]
		- [[#The Value function in this case#➳ Comparison|➳ Comparison]]
	- [[#Difference b/w Reward & Value function: Chess example#Analogy|Analogy]]
- [[#General terminology|General terminology]]


---

In **reinforcement learning (RL)**, the **reward function** and the **value function** are both critical concepts, but they serve **very different purposes**. 

# Difference b/w Reward and Value function
## ➳ Reward Function — "What do I get _now_?"
### > Definition
The **reward function** tells the agent <u>how good or bad a particular action or state</u> is — **immediately**.
- Denoted as: $R(s,a) \; or \; R(s,a,s^′)$
- It's provided by the **environment** after taking an action.
### > Purpose
To define the **goal** of the agent — it tells the agent **what it should aim to maximize**.
### > Example
If you're training a robot to pick up objects:
- **Action:** Picks up a cup.
- **Reward:** +1 if it picks it up successfully, -1 if it drops it.
    
💡 Think of the reward function as the **instant feedback signal**.

---

## ➳ Value Function — "How good is this situation _in the long run_?"
### > Definition
The **value function** estimates the <u>expected total future reward</u> the agent will get from a **state** (or state-action pair), **over time**.

- Two types:
    - **State value function**: $V(s) = \mathbb{E}[ \text{Total future reward starting from } s ]$
    - **Action value function (Q-function)**: $Q(s, a) = \mathbb{E}[ \text{Total future reward starting from } s \text{ taking action } a ]$
### > Purpose
To help the agent decide **which actions to take** to maximize future rewards — it captures **long-term consequences**.

### > Example
The robot is in front of a shelf (state $s$). It can:
- Grab a cup (action $a_1$) → might lead to more cups later.
- Knock over the shelf (action $a_2$) → might break everything.

Even if both actions give a reward of $+1$ right now, the value of action $a_2$ is lower, because future rewards will be ruined.

💡 Think of the value function as a **prediction** of long-term success.

---

##### So, What's the Key Difference?

| Concept          | Reward Function              | Value Function                                     |
| ---------------- | ---------------------------- | -------------------------------------------------- |
| **What it does** | Gives **immediate feedback** | Predicts **long-term return**                      |
| **Defined by**   | Environment                  | Learned by agent                                   |
| **Time focus**   | Instant                      | Future                                             |
| **Use case**     | Measures result of an action | Guides decision-making                             |
| **Example**      | +1 for grabbing an object    | High value if grabbing leads to more rewards later |

---

## Analogy: Playing Chess 
- **Reward Function**:  
    You get **+1 only if you win the game** at the end. All other moves = 0.
- **Value Function**:  
    Helps you estimate how **good a board position is** in terms of helping you win eventually.
    

So even if a move gives no immediate reward, it may be valuable because it **increases your chance of winning later**.

# More about "Rewards"
In RL, the reward function ==always== gives feedback <u>after each action</u>, but that <u>feedback can be delayed or sparse</u> — depending on how the environment is set up.

## What does "immediate" rewards mean?
“Immediate” doesn't always mean **non-zero** or **informative**. It just means the environment **responds immediately** with some reward, even if it's zero.

```ad-quote
💡 The reward function always provides an immediate signal — it just might be 0 (or uninformative) until a terminal condition is met.
```

## Chess example - clarified
In many chess RL setups:
- You play many moves (states + actions).
- After each move, the reward is `0`.
- Only when the game ends:
    - `+1` for a win,
    - `-1` for a loss,
    - `0` for a draw.

So yes — the environment gives a reward immediately _after each move_, but that reward is _only meaningful at the end_. Until then, the reward function returns **zero**, which is still a valid "immediate" response.


```ad-tip
The reward function is always on — it’s just **quiet sometimes** (like a judge who only claps at the end of a play).
```

---

# Difference b/w Reward & Value function: Chess example
## The Reward function in this case
### ➳ Definition

The **reward function** is a mapping:
$$
R(s_t, a_t, s_{t+1}) = 
\begin{cases}
+1 & \text{if game ends in a win at } s_{t+1} \\
-1 & \text{if game ends in a loss at } s_{t+1} \\
0 & \text{otherwise}
\end{cases}
$$


### ➳ Key characteristics
- The reward function **doesn’t guide** the agent during the game.
- It only says: "you did well" or "you didn’t" — **after the fact**.
- It's the **same for every agent** — it defines the **task**, not how to solve it.

```ad-quote
The **reward function** is _sparse_ and _delayed_, and only gives a meaningful signal **at the end** of the game.
```


## The Value function in this case
### ➳ Definition
The **value function** learns:
$$
V(s_t) = \mathbb{E}[\text{total future reward from } s_t \text{ onward}]
$$

Even though rewards only come at the end, the value function estimates how **promising a current state** is, based on the agent’s **experience**.


```ad-quote
The **value function bridges the gap** between early moves and long-delayed rewards.
```

It assigns a value to each position based on how likely it is to eventually lead to a **win** (reward = $+1$), even though there’s no immediate reward now.

---

### ➳ Comparison

|Aspect|Reward Function|Value Function|
|---|---|---|
|**Defined by**|Environment|Learned by agent|
|**When it's used**|At the end (win/loss)|During learning or planning|
|**Signal type**|Sparse (mostly 0, then +1/-1 at the end)|Smooth estimate of success|
|**Guides agent?**|No (it's delayed)|Yes (drives policy improvement)|
|**Example**|"You won the game"|"This position is 70% likely to win"|

---

## Analogy
Imagine playing a game show where you get:
- **No hints during the game**, just a final score at the end.
- You play many times.
- Over time, you **build intuition** (a value function) that says: "If I get to this part of the game, I usually win."

```ad-note
⊣⊳ That intuition = the **value function**.
	- like GPS
	- helps you find your way when you can't see the destination yet.

⊣⊳ The final score = the **reward function**.
	- is the destination (what you ultimately care about)
```


# General terminology

| Term             | Meaning                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Reward function  | Environment-defined mapping from (state, action) to a **numeric reward**. Happens **immediately after action**, even if it's zero. |
| Value function   | Estimate of expected total future rewards that the agent will get from a state over time                                           |
| Sparse reward    | A reward function that returns non-zero value only rarely, like only at the end of game.                                           |
| Immediate reward | The specific reward received **after the last action taken**. Not necessarily non-zero.                                            |
