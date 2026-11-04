PRICING = {
    "gpt-4o":          {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":     {"input": 0.15 / 1_000_000, "output":  0.60 / 1_000_000},
    "gpt-4-turbo":     {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    "claude-3-5-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-3-haiku":  {"input": 0.25 / 1_000_000, "output":  1.25 / 1_000_000},
    "claude-3-opus":   {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
}

DEFAULT_PRICING = {
    "input": 1.00 / 1_000_000,
    "output": 3.00 / 1_000_000,
}