- [[#∎ Conversion to vectors|∎ Conversion to vectors]]
	- [[#∎ Conversion to vectors#1. Numerical fields ⟶ use as-is or normalise|1. Numerical fields ⟶ use as-is or normalise]]
	- [[#∎ Conversion to vectors#2. Categorical Fields ⟶ Embed or One-Hot|2. Categorical Fields ⟶ Embed or One-Hot]]
	- [[#∎ Conversion to vectors#3. Multi-Valued Categorical Fields (Lists) ⟶ Embed & Pool|3. Multi-Valued Categorical Fields (Lists) ⟶ Embed & Pool]]
	- [[#∎ Conversion to vectors#4. Boolean or Binary Fields ⟶ 0 or 1|4. Boolean or Binary Fields ⟶ 0 or 1]]
	- [[#∎ Conversion to vectors#5. Nested Fields (Dicts) ⟶ Flatten then Encode|5. Nested Fields (Dicts) ⟶ Flatten then Encode]]
	- [[#∎ Conversion to vectors#6. Missing Values or Optional Fields?|6. Missing Values or Optional Fields?]]
	- [[#∎ Conversion to vectors#Final input vector ⟵ Concatenate all encoded fields|Final input vector ⟵ Concatenate all encoded fields]]
	- [[#∎ Conversion to vectors#Model Input ⟵ concat(context_vector, action_vector)|Model Input ⟵ concat(context_vector, action_vector)]]
	- [[#∎ Conversion to vectors#Summary|Summary]]
- [[#∎ Semantic understanding of features|∎ Semantic understanding of features]]
	- [[#∎ Semantic understanding of features#1. Semantics Come from Structure & Position|1. Semantics Come from Structure & Position]]
	- [[#∎ Semantic understanding of features#2. Learned Embeddings Preserve Semantic Relationships|2. Learned Embeddings Preserve Semantic Relationships]]
	- [[#∎ Semantic understanding of features#3. Embedding Layers Are Type-Aware by Design|3. Embedding Layers Are Type-Aware by Design]]
	- [[#∎ Semantic understanding of features#4. Deep Models Learn Interactions Across Feature Types|4. Deep Models Learn Interactions Across Feature Types]]
	- [[#∎ Semantic understanding of features#Bonus: Using Feature Encoders with Contextual Awareness|Bonus: Using Feature Encoders with Contextual Awareness]]


---

- Conversion of action and context features to uniform scale understandable by model
- How do we transform complex, heterogeneous user and item data into a uniform, learnable format for the model?
- Goal: convert all context and action fields to _fixed-dimension float vectors_. 

|Thing|What you engineer|Example|
|---|---|---|
|`Context`|✅ Context features|age, location, interests|
|`Action`|✅ Action features|topic, duration, language|
|`Context + Action`|Concatenated features|Combined input to model|

---

# ∎ Conversion to vectors
The model expects:
- `context: vector of floats` (e.g., `[0.1, 0.9, 0.3])
- `action: vector of floats` (e.g., `[1.0, 0.0, 0.4]`)

But in practice, both context and action features can look like this:
```json
context = {
  "age": 25,
  "gender": "male",
  "interests": ["dance", "comedy"],
  "location": "India",
  "time_of_day": "evening"
}

action = {
  "video_id": 12345,
  "topic": "comedy",
  "duration_sec": 14.7,
  "language": "Hindi",
  "creator_info": {
    "followers": 100000,
    "verified": true
  }
}
```


## 1. Numerical fields ⟶ use as-is or normalise
- `age` → use directly or scale (`age / 100`)
- `duration_sec` → normalize (`duration / 60`)
- `followers` → log-transform (`log(1 + followers)`)

Tip: Normalise each numeric field to roughly [0, 1] range.


## 2. Categorical Fields ⟶ Embed or One-Hot
For example:
- `gender = "male"` → One-hot → `[1, 0]`
- `language = "Hindi"` → Embed → `embedding_lookup("Hindi") → [0.3, 0.1, ...]`

You typically use **embedding layers** for high-cardinality fields like:
- `topic`, `location`, `language`, `creator_id`, etc.

Example:
```python
topic_embedding = nn.Embedding(num_topics, emb_dim)
language_embedding = nn.Embedding(num_langs, emb_dim)
```
These layers are learned _just like weights_, and they turn strings into dense, learnable float vectors.


## 3. Multi-Valued Categorical Fields (Lists) ⟶ Embed & Pool

- `interests = ["dance", "comedy"]`
- Map each to embedding: `[e1, e2]`
- Apply mean pooling / max pooling: `avg([e1, e2])`

This gives you a single vector representing the whole list of interests.


## 4. Boolean or Binary Fields ⟶ 0 or 1
- `verified: true` → `1`
- `verified: false` → `0`

Easy and directly usable.


## 5. Nested Fields (Dicts) ⟶ Flatten then Encode
- `creator_info = { "followers": 1e5, "verified": true }`
    - Extract `followers → log(f + 1)`
    - Extract `verified → 0 or 1`

Basically treat sub-fields like regular features.


## 6. Missing Values or Optional Fields?
Options:
- Add a “missing” token to embeddings
- Impute default value (e.g., `0`)
- Add a mask (0/1) feature saying whether it was present

## Final input vector ⟵ Concatenate all encoded fields
You convert:
```python
context = {
  age, gender, interests, location, ...
}
```
into:
```python
context_vector = concat(
  norm_age, onehot_gender, pooled_interest_embeddings, loc_embedding, ...
)
```

Same with action:
```python
action_vector = concat(
  topic_embedding, norm_duration, lang_embedding, log_followers, ...
)
```

## Model Input ⟵ concat(context_vector, action_vector)
Just like this:
```python
x = torch.cat([context_vector, action_vector], dim=-1)
output = model(x)  # predict reward
```

## Summary

| Data Type         | How to Handle                                |
| ----------------- | -------------------------------------------- |
| `int/float`       | Normalize / log-scale                        |
| `string`          | Embed or one-hot                             |
| `list of strings` | Embed + pool                                 |
| `bool`            | Convert to 0/1                               |
| `dict`            | Flatten and encode each field                |
| `missing`         | Use default values or add missing indicators |
Map every input to a fixed-length float vector, so you can feed it into your model.

# ∎ Semantic understanding of features
When we convert categorical/textual fields into embeddings or floats, it’s true that their original “human-readable meaning” is not obvious anymore. But, the model can still learn their meaning through their position, structure, and they are encoded.

## 1. Semantics Come from Structure & Position
Even after converting everything to float vectors, you keep **semantic information** because:
- `context_feature_1` is **always** "age"
- `context_feature_2` is **always** "gender"
- `action_feature_3` is **always** "video_topic"

The model learns:
> “Whatever appears in slot 3 always means video topic” — so it can assign semantics to the slot itself.

This is why **positional consistency** is so important.

## 2. Learned Embeddings Preserve Semantic Relationships

Example:
- `"music"` and `"dance"` both get embeddings like:
    - `music → [0.21, 0.93, 0.15]`
    - `dance → [0.23, 0.90, 0.18]`

Why? Because they co-occur with similar users, contexts, and get similar rewards.
The model learns to put semantically similar tokens close in space, even though it doesn't "know" what music is.

That's **semantics emerging from data**.


## 3. Embedding Layers Are Type-Aware by Design
When you do:
```python
self.gender_embedding = nn.Embedding(num_genders, dim)
self.language_embedding = nn.Embedding(num_langs, dim)
```

The model knows:
> “This embedding table is just for gender.”  
> “This other one is just for language.”

So even though they both become `[dim]`-dimensional float vectors, the embedding table itself provides type-awareness.

Type semantics are encoded in which table and where the value came from.


## 4. Deep Models Learn Interactions Across Feature Types
Suppose your model sees:

|Age|Gender|Topic|Reward|
|---|---|---|---|
|25|Male|Dance|1|
|25|Male|Politics|0|
|40|Female|Dance|0|
|20|Female|Dance|1|

It will learn:
- `"Dance"` works for young users
- `"Politics"` doesn’t work for that age/gender group

Even though everything is embedded as floats — the structure and reward feedback give it contextual meaning.

Semantics are learned through **correlations with outcomes**.


## Bonus: Using Feature Encoders with Contextual Awareness
If you want to go even further, you can:
- Add **field-aware encoders**: where each field has a separate transformation
- Use **attention mechanisms**: to let the model learn which parts of the context are most relevant to each action
- Use **feature type embeddings**: like adding a "this is a 'user_interest' field" token to the model

Even though everything becomes a float vector, you don't lose semantics, because:
- Structure and position keep the **meaning of each slot**
- Embedding tables are **field-specific**
- Semantics emerge from **co-occurrence, correlation with reward**
- The model **learns type interactions** through training
