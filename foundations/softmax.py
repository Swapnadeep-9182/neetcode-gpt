import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_shifted = z - np.max(z)
        natural_exp = np.exp(z_shifted)
        total_sum = natural_exp.sum()
        ans = natural_exp/total_sum
        return np.round(ans,4)