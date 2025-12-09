"""
Utility functions for CDE code evaluation environment.
"""

import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard imports to prepend to all code (matching original project)
BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase, ascii_uppercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt, factorial, atan2, pi
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge, nlargest, nsmallest, heapreplace
from functools import reduce, cache, lru_cache, cmp_to_key, reduce
from random import randrange, shuffle
from operator import itemgetter, sub, xor, or_
from re import search as re_search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator, Deque
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import datetime
from time import time
import numpy as np
import pandas as pd
from math import log, prod
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle, pairwise
from functools import lru_cache, reduce, partial
from operator import iand
import sys
import io, os
"""


def generate_function_wrapper(
    user_code: str,
    fn_name: str,
    test_input,
    base_imports: str = BASE_IMPORTS
) -> str:
    """
    Generate a Python script that wraps user code for function-based testing.
    
    Args:
        user_code: The user's solution code (should already include BASE_IMPORTS)
        fn_name: Name of the function to test
        test_input: Test input - can be:
                    - A string with newline-separated values (needs parsing)
                    - A Python object (already parsed)
        base_imports: Standard imports (not used, kept for compatibility)
    
    Returns:
        Complete Python script for testing
    """
    # Parse test_input following original project logic
    # If test_input is a string, it contains newline-separated parameter values
    # that need to be eval'd. Otherwise, it's already a parsed object.
    if isinstance(test_input, str):
        # Split by newline and eval each line (matching original project)
        try:
            inputs_repr = repr(list(map(eval, test_input.split("\n"))))
        except Exception:
            # Fallback: treat as single argument
            inputs_repr = repr([test_input])
    elif isinstance(test_input, list):
        # Already a list of arguments
        inputs_repr = repr(test_input)
    else:
        # Single argument
        inputs_repr = repr([test_input])
    
    wrapper = f"""
{user_code}

import json
try:
    inputs = {inputs_repr}
    if "Solution" in locals() or "Solution" in globals():
        solution_instance = Solution()
        result = getattr(solution_instance, "{fn_name}")(*inputs)
    else:
        result = {fn_name}(*inputs)
    print(json.dumps({{"success": True, "result": result}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": repr(e)}}))
"""
    return wrapper


def compare_stdout_results(actual: str, expected: str) -> bool:
    """
    Compare two stdout results with multiple strategies for flexibility.
    
    Strategies (in order):
    1. Exact match after stripping whitespace
    2. Line-by-line comparison (ignoring extra whitespace)
    3. Token-by-token comparison
    4. Numeric comparison with tolerance
    
    Args:
        actual: Actual output from code execution
        expected: Expected output
    
    Returns:
        True if outputs match by any strategy
    """
    # Strategy 1: Strip and compare
    if actual.strip() == expected.strip():
        return True
    
    # Strategy 2: Line-wise comparison
    if _compare_linewise(actual, expected):
        return True
    
    # Strategy 3: Token-wise comparison
    if _compare_tokenwise(actual, expected):
        return True
    
    # Strategy 4: Numeric comparison with tolerance
    if _compare_numeric_tokens(actual, expected):
        return True
    
    return False


def _compare_linewise(actual: str, expected: str) -> bool:
    """Compare outputs line by line, ignoring extra whitespace."""
    actual_lines = [line.strip() for line in actual.strip().split('\n')]
    expected_lines = [line.strip() for line in expected.strip().split('\n')]
    
    if len(actual_lines) != len(expected_lines):
        return False
    
    return all(a == e for a, e in zip(actual_lines, expected_lines))


def _compare_tokenwise(actual: str, expected: str) -> bool:
    """Compare outputs token by token (split by whitespace)."""
    actual_tokens = actual.strip().split()
    expected_tokens = expected.strip().split()
    
    if len(actual_tokens) != len(expected_tokens):
        return False
    
    return all(a == e for a, e in zip(actual_tokens, expected_tokens))


def _compare_numeric_tokens(actual: str, expected: str, tolerance: float = 1e-3) -> bool:
    """
    Compare outputs treating all tokens as potential numbers with tolerance.
    
    Args:
        actual: Actual output
        expected: Expected output
        tolerance: Tolerance for floating point comparison
    
    Returns:
        True if all numeric tokens match within tolerance
    """
    actual_tokens = actual.strip().split()
    expected_tokens = expected.strip().split()
    
    if len(actual_tokens) != len(expected_tokens):
        return False
    
    for a_token, e_token in zip(actual_tokens, expected_tokens):
        # Try numeric comparison
        try:
            a_num = Decimal(a_token)
            e_num = Decimal(e_token)
            if abs(a_num - e_num) > Decimal(str(tolerance)):
                return False
        except (InvalidOperation, ValueError):
            # Not a number, must match exactly
            if a_token != e_token:
                return False
    
    return True


def compare_function_results(actual: Any, expected: Any) -> bool:
    """
    Compare function call results with special handling for different types.
    
    Handles:
    - Tuple/list equivalence
    - Nested structures
    - Single-element list wrapping
    
    Args:
        actual: Actual result from function call
        expected: Expected result
    
    Returns:
        True if results are equivalent
    """
    # Direct equality
    if actual == expected:
        return True
    
    # Handle tuple/list equivalence
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(compare_function_results(a, e) for a, e in zip(actual, expected))
    
    # Handle single-element list wrapping (common in LeetCode)
    if isinstance(actual, list) and len(actual) == 1:
        return compare_function_results(actual[0], expected)
    if isinstance(expected, list) and len(expected) == 1:
        return compare_function_results(actual, expected[0])
    
    # Handle string representations of booleans
    if isinstance(actual, str) and isinstance(expected, str):
        actual_lower = actual.lower().strip()
        expected_lower = expected.lower().strip()
        if actual_lower in ('true', 'false') and expected_lower in ('true', 'false'):
            return actual_lower == expected_lower
    
    # Handle numeric comparison with tolerance
    try:
        a_num = Decimal(str(actual))
        e_num = Decimal(str(expected))
        return abs(a_num - e_num) < Decimal('1e-6')
    except (InvalidOperation, ValueError, TypeError):
        pass
    
    return False