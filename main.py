from ciphertext_env import CipherEnv
from agent import HonestAgent
from lm import NgramLM
from run_episode import run_episode

lm = NgramLM()
env = CipherEnv(text_len = 80, max_rounds = 10)

vocab = ["THE", "WAS,", "HE", "SHE", "THAT", "THERE", "HAD", "AND", "OR", "NOT", "IS", "WANT"]

agents = [
    HonestAgent("A1", vocab, env.text_len, lm),
    HonestAgent("A2", vocab, env.text_len, lm),
    HonestAgent("A3", vocab, env.text_len, lm)
]

run_episode(env, agents, lm)