# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import Counter
from typing import List, Union, Any
from math import log2
from enum import Enum
import random



class ComparisonMode(Enum):
    EACH_WITH_EACH = 1
    INDEX = 2
    ASYMMETRIC_SUBSET = 3
    ASYMMETRIC_PATTERN = 4
    ASYMMETRIC_WEIGHTED = 5
    ASYMMETRIC_SLIDING_WINDOW = 6
    ASYMMETRIC_ONE_SIDED = 7
    ASYMMETRIC_PROPORTIONAL = 8
    ASYMMETRIC_THRESHOLD = 9
    ASYMMETRIC_MULTI_ELEMENT = 10
    ASYMMETRIC_RANDOM = 11
    ASYMMETRIC_CUMULATIVE = 12
    ASYMMETRIC_EXPANSION = 13


class SimilarityCalculator:
    def __init__(
        self,
        A: List[Union[str, float]],
        B: List[Union[str, float]],
        overlap_weight: float = 0.075,
        length_weight: float = 0.025,
        numerical_weight: float = 0.9,
        string_weight: float = 0.9,
        comparison_mode: ComparisonMode = ComparisonMode.EACH_WITH_EACH,  # Default comparison mode
        threshold: float = 0.1  # Used for threshold-based comparisons
    ):
        """
        Initialize SimilarityCalculator with two lists, corresponding weights, and comparison mode.
        """
        self._A = A
        self._B = B
        self._overlap_weight = overlap_weight
        self._length_weight = length_weight
        self._numerical_weight = numerical_weight
        self._string_weight = string_weight
        self._comparison_mode = comparison_mode
        self._threshold = threshold

    def entropy(self, s: str) -> float:
        """
        Calculate the Shannon entropy of a string.
        """
        total_len = len(s)
        if total_len == 0:
            return 0.0

        freq = Counter(s)
        probs = [count / total_len for count in freq.values()]
        return -sum(p * log2(p) for p in probs if p > 0)

    def character_overlap(self, A: str, B: str) -> float:
        """
        Compute the character overlap between two strings.
        """
        counter_A = Counter(A)
        counter_B = Counter(B)
        shared_chars = counter_A & counter_B
        total_shared = sum(shared_chars.values())
        max_len = max(len(A), len(B))
        return total_shared / max_len if max_len > 0 else 0.0

    def number_overlap(self, A: List[float], B: List[float]) -> float:
        """
        Compute the numerical overlap between two lists of numbers.
        """
        counter_A = Counter(A)
        counter_B = Counter(B)
        shared_numbers = counter_A & counter_B
        total_shared = sum(shared_numbers.values())
        max_len = max(len(A), len(B))
        return total_shared / max_len if max_len > 0 else 0.0

    def numerical_similarity(self, A: List[float], B: List[float]) -> float:
        """
        Compute the similarity between two lists of numbers.
        """
        if not A or not B:
            return 0.0

        A = [float(a) for a in A]
        B = [float(b) for b in B]
        min_len = min(len(A), len(B))

        total_diff = sum(abs(A[i] - B[i]) for i in range(min_len))
        max_sum = sum(max(abs(A[i]), abs(B[i])) for i in range(min_len))

        # Account for extra elements in longer list
        if len(A) > min_len:
            total_diff += sum(abs(a) for a in A[min_len:])
            max_sum += sum(abs(a) for a in A[min_len:])
        if len(B) > min_len:
            total_diff += sum(abs(b) for b in B[min_len:])
            max_sum += sum(abs(b) for b in B[min_len:])

        if max_sum == 0:
            return 1.0  # Both lists are zeros

        similarity = 1 - (total_diff / max_sum)
        return max(similarity, 0.0)

    def length_normalization(self, A: List[Any], B: List[Any]) -> float:
        """
        Normalize similarity based on the lengths of two lists.
        """
        len_A = len(A)
        len_B = len(B)

        if len_A == 0 and len_B == 0:
            return 1.0
        if len_A == 0 or len_B == 0:
            return 0.0

        return 1 - (abs(len_A - len_B) / max(len_A, len_B))

    def mixed_similarity(self) -> float:
        """
        Compute the similarity between two lists containing mixed types (numbers and strings).
        """
        A = self._A
        B = self._B

        if A == B:
            return 1.0

        def is_numeric(x):
            return isinstance(x, (int, float))

        def is_string(x):
            return isinstance(x, str)

        # Separate numeric and string elements
        A_nums = [x for x in A if is_numeric(x)]
        B_nums = [x for x in B if is_numeric(x)]
        A_strs = [x for x in A if is_string(x)]
        B_strs = [x for x in B if is_string(x)]

        # Numerical similarity
        num_sim = self.numerical_similarity(A_nums, B_nums) if A_nums and B_nums else 0.0
        num_overlap = self.number_overlap(A_nums, B_nums) if A_nums and B_nums else 0.0

        # String similarity
        if A_strs and B_strs:
            # Concatenate strings for comparison
            A_str_concat = ''.join(A_strs)
            B_str_concat = ''.join(B_strs)
            entropy_a = self.entropy(A_str_concat)
            entropy_b = self.entropy(B_str_concat)
            overlap = self.character_overlap(A_str_concat, B_str_concat)
            entropy_diff = abs(entropy_a - entropy_b) / max(entropy_a, entropy_b, 1e-9)
            string_sim = overlap * (1 - entropy_diff)
        else:
            string_sim = 0.0
            overlap = 0.0

        # Length normalization
        length_norm = self.length_normalization(A, B)

        # Combine similarities with weights
        total_similarity = (
            num_sim * self._numerical_weight +
            string_sim * self._string_weight +
            length_norm * self._length_weight +
            overlap * self._overlap_weight
        )

        # Normalize total weights
        total_weights = self._numerical_weight + self._string_weight + self._length_weight + self._overlap_weight
        final_similarity = total_similarity / total_weights

        # Ensure the similarity is within [0,1]
        return max(0.0, min(1.0, final_similarity))

    def asymmetric_similarity(self) -> float:
        """
        Handle different comparison modes and asymmetric comparisons.
        """
        A = self._A
        B = self._B
        mode = self._comparison_mode

        if mode == ComparisonMode.EACH_WITH_EACH:
            return self.mixed_similarity()  # Already covers each-with-each and index

        elif mode == ComparisonMode.INDEX:
            return self.mixed_similarity()  # Index is the default

        elif mode == ComparisonMode.ASYMMETRIC_SUBSET:
            # Compare first half of A with B
            subset_A = A[:len(A) // 2]
            return SimilarityCalculator(subset_A, B).mixed_similarity()

        elif mode == ComparisonMode.ASYMMETRIC_PATTERN:
            # Compare elements based on a pattern (every second element)
            pattern_A = A[::2]
            pattern_B = B[::2]
            return SimilarityCalculator(pattern_A, pattern_B).mixed_similarity()

        elif mode == ComparisonMode.ASYMMETRIC_WEIGHTED:
            # Weighted comparison (e.g., first element has double weight)
            weights = [2.0] + [1.0] * (len(A) - 1)
            weighted_A = [a * weights[i] for i, a in enumerate(A) if isinstance(a, (int, float))]
            return SimilarityCalculator(weighted_A, B).mixed_similarity()

        elif mode == ComparisonMode.ASYMMETRIC_SLIDING_WINDOW:
            # Sliding window comparison: filter only numeric values
            window_size = 2
            A_numeric = [a for a in A if isinstance(a, (int, float))]
            if len(A_numeric) < window_size:
                return 0.0  # Not enough numeric elements for window comparison
            A_window = [sum(A_numeric[i:i + window_size]) for i in range(0, len(A_numeric) - window_size + 1)]
            return SimilarityCalculator(A_window, B).mixed_similarity()

        elif mode == ComparisonMode.ASYMMETRIC_ONE_SIDED:
            # Compare each element in A with the full list B
            total_similarity = 0
            for a in A:
                total_similarity += SimilarityCalculator([a], B).mixed_similarity()
            return total_similarity / len(A)

        elif mode == ComparisonMode.ASYMMETRIC_PROPORTIONAL:
            # Proportional comparison based on list lengths
            scale_factor = len(B) / len(A) if len(A) < len(B) else len(A) / len(B)
            scaled_A = [A[int(i / scale_factor)] for i in range(len(B))]
            return SimilarityCalculator(scaled_A, B).mixed_similarity()

        elif mode == ComparisonMode.ASYMMETRIC_THRESHOLD:
            # Only compare elements if their difference exceeds a threshold
            filtered_A = [a for a, b in zip(A, B) if isinstance(a, (int, float)) and abs(a - b) > self._threshold]
            filtered_B = [b for a, b in zip(A, B) if isinstance(b, (int, float)) and abs(a - b) > self._threshold]
            return SimilarityCalculator(filtered_A, filtered_B).mixed_similarity()

        elif mode == ComparisonMode.ASYMMETRIC_MULTI_ELEMENT:
            # Compare each element of A with multiple elements of B
            total_similarity = 0
            for i in range(len(A)):
                for j in range(min(len(B), i + 2)):
                    total_similarity += SimilarityCalculator([A[i]], [B[j]]).mixed_similarity()
            return total_similarity / (len(A) * 2)

        elif mode == ComparisonMode.ASYMMETRIC_RANDOM:
            # Random comparison between elements
            random_A = [A[random.randint(0, len(A) - 1)] for _ in range(len(B))]
            return SimilarityCalculator(random_A, B).mixed_similarity()


        elif mode == ComparisonMode.ASYMMETRIC_CUMULATIVE:
            # Cumulative comparison: only sum numeric elements
            A_numeric = [a for a in A if isinstance(a, (int, float))]
            B_numeric = [b for b in B if isinstance(b, (int, float))]

            if not A_numeric or not B_numeric:
                return 0.0  # If there are no numeric elements, return 0

            cumulative_A = [sum(A_numeric[:i + 1]) for i in range(len(A_numeric))]
            cumulative_B = [sum(B_numeric[:i + 1]) for i in range(len(B_numeric))]
            return SimilarityCalculator(cumulative_A, cumulative_B).mixed_similarity()



        elif mode == ComparisonMode.ASYMMETRIC_EXPANSION:
            # Expand each element in A and compare with B
            expanded_A = [a for a in A for _ in range(2)]  # Duplicate each element
            return SimilarityCalculator(expanded_A, B).mixed_similarity()

        else:
            return 0.0  # Default case


# Example Usage
def main():
    A = [1.2, 'abc', 16.0]
    B = [0, 'ab', 15]

    # Initialize with each-with-each mode
    calc = SimilarityCalculator(A, B, comparison_mode=ComparisonMode.EACH_WITH_EACH)

    # Get similarity result for EACH_WITH_EACH mode
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (EACH_WITH_EACH): {similarity}")

    # Switch to INDEX comparison mode
    calc._comparison_mode = ComparisonMode.INDEX
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (INDEX): {similarity}")

    # Switch to ASYMMETRIC_SUBSET comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_SUBSET
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_SUBSET): {similarity}")

    # Switch to ASYMMETRIC_PATTERN comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_PATTERN
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_PATTERN): {similarity}")

    # Switch to ASYMMETRIC_WEIGHTED comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_WEIGHTED
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_WEIGHTED): {similarity}")

    # Switch to ASYMMETRIC_SLIDING_WINDOW comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_SLIDING_WINDOW
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_SLIDING_WINDOW): {similarity}")

    # Switch to ASYMMETRIC_ONE_SIDED comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_ONE_SIDED
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_ONE_SIDED): {similarity}")

    # Switch to ASYMMETRIC_PROPORTIONAL comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_PROPORTIONAL
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_PROPORTIONAL): {similarity}")

    # Switch to ASYMMETRIC_THRESHOLD comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_THRESHOLD
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_THRESHOLD): {similarity}")

    # Switch to ASYMMETRIC_MULTI_ELEMENT comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_MULTI_ELEMENT
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_MULTI_ELEMENT): {similarity}")

    # Switch to ASYMMETRIC_RANDOM comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_RANDOM
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_RANDOM): {similarity}")

    # Switch to ASYMMETRIC_CUMULATIVE comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_CUMULATIVE
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_CUMULATIVE): {similarity}")

    # Switch to ASYMMETRIC_EXPANSION comparison mode
    calc._comparison_mode = ComparisonMode.ASYMMETRIC_EXPANSION
    similarity = calc.asymmetric_similarity()
    print(f"Similarity between A and B (ASYMMETRIC_EXPANSION): {similarity}")


if __name__ == "__main__":
    main()
