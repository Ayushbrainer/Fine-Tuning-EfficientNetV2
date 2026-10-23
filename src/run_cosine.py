"""Section 4: LR=0.05, 300 epochs, per-batch cosine annealing."""
from common import run

run("cosine", lr=0.05, epochs=300, cosine=True)
