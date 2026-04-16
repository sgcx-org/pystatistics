"""
Multinomial logistic regression (softmax regression).

Matches R's nnet::multinom() for the multinomial logit model.

Usage:
    from pystatistics.multinomial import multinom, MultinomialSolution

    result = multinom(y, X)
    result.summary()
"""

from pystatistics.multinomial._solver import multinom
from pystatistics.multinomial.solution import MultinomialSolution

__all__ = [
    "multinom",
    "MultinomialSolution",
]
