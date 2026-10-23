"""Section 4: LR=0.05, 300 epochs, no cosine annealing."""
from common import run

run("nocosine", lr=0.05, epochs=300)
