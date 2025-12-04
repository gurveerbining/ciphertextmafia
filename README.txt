
Mafia Ciphertext Game — Multi-Agent RL Crib-Dragging Decoder
============================================================

Overview
--------
This project implements a multi-agent reinforcement learning (RL) system designed to collaboratively break a two-time-pad ciphertext pair using crib dragging. Two honest agents attempt to reconstruct the plaintexts, while an imposter agent attempts to subtly disrupt progress.

The system simulates proposed “cribs” (candidate plaintext segments) placed at different offsets of two ciphertexts. Honest agents vote based on a heuristic scoring system that evaluates whether a proposal looks linguistically valid and consistent with what has already been revealed. The environment updates the plaintext masks when a correct crib helps recover new characters.

The RL component reinforces agents for making progress, while the heuristic scorer acts as the linguistic model guiding acceptance/rejection of proposals.

Core Components
---------------
1. **ciphertext_env.py**
   - Generates plaintexts from Brown Corpus samples.
   - Encrypts using two-time-pad (same XOR key for both messages).
   - Maintains masks for revealed plaintext characters.
   - Applies proposals and returns rewards + completion status.

2. **agent.py**
   - `HonestAgent`: RL-based agent with Q-table, epsilon-greedy action selection, and trust tracking.
   - `ImposterAgent`: Attempts to interfere by occasionally flipping votes or proposing bad cribs.
   - Agents vote using the heuristic scorer rather than a language model.

3. **heuristics.py**
   - Heuristic scoring model that evaluates:
     - Mask consistency (no contradictions with revealed plaintext)
     - Character plausibility (A–Z, spaces)
     - Common English bigrams/trigrams (“TH”, “THE”, “ING”, etc.)
     - Avoiding unrealistic patterns (double spaces)
     - Reward proportional to how many unknown characters a crib would reveal

4. **run_episode.py**
   - Simulates one full multi-round episode of proposals, voting, environment updates, and early stopping when plaintexts are sufficiently recovered.

5. **main.py**
   - Runs multiple episodes.
   - Logs completion rates for performance analysis.

Heuristic Scoring Summary
-------------------------
The heuristic scorer acts as a lightweight language model substitute:

- **+3** for matching a known plaintext character
- **−5** for contradicting a known character
- **+1** for uppercase letters
- **+0.5** for spaces
- **−2** for invalid characters
- **+1** for each common English bigram found
- **+2** for each common trigram found
- **−4** for double spaces
- **+4 per newly revealed character** (strong reward for progress)

Normalized by crib length.

This scoring enables the honest agents to prefer realistic proposals while rejecting garbage input.

Learning Behavior
-----------------
Although the heuristics heavily shape which proposals get accepted, RL still occurs:

- The Q-table learns which (side, offset, crib) combinations historically produced high rewards.
- Epsilon decay shifts behavior from exploration -> exploitation.
- Completion rate trends typically rise over many episodes, indicating improved strategy.

Results
-------
Typical completion rates fall between **75–90%**, depending on plaintext length and available crib set.

Early stopping triggers automatically when enough characters are revealed.

Usage
-----
Run the main:

    python main.py

To log output to a file:

    python main.py > output.txt

Repository Structure
--------------------
- `main.py`
- `run_episode.py`
- `ciphertext_env.py`
- `agent.py`
- `heuristics.py`
- `README.txt`
- `output.txt`

