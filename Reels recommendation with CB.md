- [[#∎ Problem setting|∎ Problem setting]]
	- [[#∎ Problem setting#➳ Context $(x_t)$ — The User State|➳ Context $(x_t)$ — The User State]]
	- [[#∎ Problem setting#➳ Actions $(a_t)$ — The Reels You Could Recommend|➳ Actions $(a_t)$ — The Reels You Could Recommend]]
	- [[#∎ Problem setting#➳ Reward $(r_t)$ — Feedback on the Shown Reel|➳ Reward $(r_t)$ — Feedback on the Shown Reel]]
	- [[#∎ Problem setting#➳ The Bandit task at time $t$:|➳ The Bandit task at time $t$:]]
	- [[#∎ Problem setting#➳ Scale with millions of actions?|➳ Scale with millions of actions?]]
- [[#∎ Toy example|∎ Toy example]]
	- [[#∎ Toy example#➳ Step:1 - Mini code snippet|➳ Step:1 - Mini code snippet]]
	- [[#∎ Toy example#➳ Step:2 - Neural Policy Network for CB|➳ Step:2 - Neural Policy Network for CB]]
	- [[#∎ Toy example#➳ Step:3 - Exploration in CBs|➳ Step:3 - Exploration in CBs]]
		- [[#➳ Step:3 - Exploration in CBs#1. Epsilon-Greedy|1. Epsilon-Greedy]]
		- [[#➳ Step:3 - Exploration in CBs#2. Thompson Sampling|2. Thompson Sampling]]
		- [[#➳ Step:3 - Exploration in CBs#3. Upper Confidence Bound (UCB)|3. Upper Confidence Bound (UCB)]]
- [[#∎ Appendix|∎ Appendix]]

---
# ∎ Problem setting
## ➳ Context $(x_t)$ — The User State
This is **everything you know about the user** at the moment of decision. It guides what kind of content they might like.

Examples of context features:
- `age`: 24
- `gender`: male
- `location`: Mumbai
- `interests`: [dance, music, tech]
- `device_type`: iPhone
- `time_of_day`: evening
- `recent activity`: watched 3 dance reels
- `session duration so far`: 8 minutes

```ad-info
<i><u>Context is what varies with each decision.</u></i>
```


## ➳ Actions $(a_t)$ — The Reels You Could Recommend
Each **action** is a **specific video (reel)** that you could recommend.
- You have millions of videos, but maybe only sample or rank a smaller subset per user (to make the problem tractable).
- Each video also has its own metadata — we call these ==action features==.

Examples of action features:
- `video_id`: 81234
- `creator`: user_567
- `topic`: dance
- `duration`: 15 sec
- `engagement_score`: 0.8
- `language`: Hindi
- `audio_type`: music

```ad-info
<i><u>These features help generalize across videos (e.g., even if a user hasn’t seen this video before, we can predict if they’ll like it).</u></i>
```


## ➳ Reward $(r_t)$ — Feedback on the Shown Reel
You only get feedback for the **one reel you show**.

**Possible reward definitions:**
- Binary: `1` if the user watches it fully, `0` otherwise.
- Probabilistic: Probability the user shares/likes/saves.
- Continuous: Actual **watch time** normalized.
- Composite: A weighted mix of different actions (watch + like + share).

```ad-info
<i><u>You use this reward to update your policy.</u></i>
```

| Component       | in reel setting                                   |
| --------------- | ------------------------------------------------- |
| context ($x_t$) | user info + session info                          |
| actions ($a_t$) | reels/videos we can show                          |
| action features | metadata about each video (topic, duration, etc.) |
| reward ($r_t$)  | whether the user watched, liked, or engaged       |
 
## ➳ The Bandit task at time $t$:
1. Observe user `u_t` (context `x_t`)
2. Select a video `a_t` (from a pool of candidates)
3. Show it to the user
4. Observe reward `r_t` (e.g., watch time)
5. Update your model so it improves future choices


## ➳ Scale with millions of actions?
- Use **candidate generation** to shortlist top-N reels.
- Learn a **policy model** that ranks those candidates based on `(context, action features)`.
- Algorithms like **LinUCB**, **Thompson Sampling**, or **policy gradient bandits** can be used.

---

# ∎ Toy example
## ➳ Step:1 - Mini code snippet
We’ll simulate a **contextual bandit** with 2 users, 3 reels, and fake rewards.

```python
import numpy as np

# Simulated user contexts (e.g., age, location, preference scores)
user_contexts = {
    "user_1": np.array([0.8, 0.1, 0.3]),  # Likes music
    "user_2": np.array([0.2, 0.9, 0.4])   # Likes politics
}

# Simulated video action features (e.g., is_music, is_politics, duration)
video_features = {
    "reel_1": np.array([1, 0, 0.2]),  # Music
    "reel_2": np.array([0, 1, 0.3]),  # Politics
    "reel_3": np.array([0, 0, 0.5])   # Generic
}

# Simple dot-product scoring function (Linear bandit)
def score(context, action_feat):
    return np.dot(context, action_feat)

# Agent picks best reel based on score
def choose_action(context):
    scores = {k: score(context, v) for k, v in video_features.items()}
    return max(scores.items(), key=lambda x: x[1])  # (reel, score)

# Simulate reward: watch if score > threshold
def get_reward(score):
    return 1 if score > 0.3 else 0

# Run for user_1
context = user_contexts["user_1"]
chosen_reel, s = choose_action(context)
reward = get_reward(s)

print(f"User saw: {chosen_reel}, score: {s:.2f}, reward: {reward}")
```

This is a linear bandit — next step: neural version ↓

## ➳ Step:2 - Neural Policy Network for CB
Now we’ll model the interaction between context and action via a neural network.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Dimensions
context_dim = 3   # e.g., age, preference scores
action_dim = 3    # e.g., video tags
hidden_dim = 16

# Input = [context || action] concatenated
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Predict expected reward
            nn.Sigmoid()               # Normalize to [0, 1]
        )

    def forward(self, context, action):
        x = torch.cat([context, action], dim=-1)
        return self.net(x)

# Training loop (pseudo)
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

for step in range(1000):
    context = torch.tensor([[0.9, 0.1, 0.3]])         # sample user
    action = torch.tensor([[1.0, 0.0, 0.2]])          # sample video
    pred = policy(context, action)                    # predicted reward
    actual_reward = torch.tensor([[1.0]])             # simulated ground truth

    loss = nn.BCELoss()(pred, actual_reward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
This model learns a function `f(context, action) → expected reward`.

## ➳ Step:3 - Exploration in CBs
Bandits need to explore to avoid being biased early on. Common strategies:
### 1. Epsilon-Greedy
- With probability `ε`, choose a **random** video (explore).
- With probability `1 - ε`, choose the **best** video (exploit).
### 2. Thompson Sampling
- Sample model weights from a posterior distribution.
- Choose the action that’s best under that sample.
- Encourages actions with **high uncertainty**.
### 3. Upper Confidence Bound (UCB)
- Score = `mean reward + uncertainty`
- Chooses actions with high **potential** (like optimism).

Example with Epsilon-Greedy:
```python
import random

def choose_with_epsilon(context, epsilon=0.2):
    if random.random() < epsilon:
        return random.choice(list(video_features.items()))  # Explore
    else:
        return choose_action(context)  # Exploit
```


---

# ∎ Appendix

```ad-question
#### Q. Difference between action features and context?
Ans.

They're *separate, but compatible*.

|Type|What it describes|Example|
|---|---|---|
|Context|User state|Age, preferences, location|
|Action features|Video/item features|Topic, duration, tags, language|

But they are often combined together when fed into a model:
`input = concat(context, action)  # Final input shape = context_dim + action_dim`

So, the model learns:
$$f(context, actions) \approx \text{expected reward}$$

This let's us generalize accoss users and videos.
```

