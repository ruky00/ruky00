"""
Compact finance sentiment lexicon (Loughran-McDonald-inspired subset).

The full Loughran-McDonald master dictionary has ~2,300 negative and ~350
positive terms tuned for financial text. Embedding all of it here would be
bulky; this is a curated, readable subset that captures the most common
headline language. Extend the sets — or load the full LM CSV — for production.

Words are lower-case stems matched against tokenised headlines.
"""

NEGATIVE = {
    # losses / decline
    "loss", "losses", "loses", "lost", "decline", "declines", "declined",
    "fall", "falls", "fell", "drop", "drops", "dropped", "plunge", "plunges",
    "plunged", "slump", "slumps", "tumble", "tumbles", "tumbled", "sink",
    "sinks", "slide", "slides", "crash", "crashes", "crashed", "collapse",
    "collapses", "downturn", "selloff", "sell-off", "rout", "weak", "weakness",
    "weaker", "sluggish", "soft", "softer", "miss", "misses", "missed",
    "shortfall", "disappointing", "disappoint", "disappointed", "warn",
    "warns", "warning", "warned", "cut", "cuts", "cutting", "slash", "slashes",
    "slashed", "downgrade", "downgrades", "downgraded", "lower", "lowered",
    "negative", "bearish", "recession", "fear", "fears", "concern", "concerns",
    "worried", "worry", "risk", "risks", "risky", "volatile", "volatility",
    # legal / governance trouble
    "lawsuit", "lawsuits", "sue", "sued", "fraud", "fraudulent", "probe",
    "investigation", "investigated", "scandal", "fine", "fined", "penalty",
    "penalties", "sanction", "sanctions", "breach", "default", "defaults",
    "bankruptcy", "bankrupt", "insolvent", "insolvency", "delist", "delisted",
    "recall", "recalls", "halt", "halted", "suspend", "suspended", "layoff",
    "layoffs", "fired", "resign", "resigns", "resigned", "ousted",
    # macro / demand
    "inflation", "deflation", "slowdown", "contraction", "deficit", "debt",
    "crisis", "turmoil", "uncertainty", "headwind", "headwinds", "pressure",
    "pressured", "struggle", "struggles", "struggling", "underperform",
    "underperforms", "underperformed", "glut", "oversupply", "writedown",
    "write-down", "impairment", "deteriorate", "deteriorating",
}

POSITIVE = {
    "gain", "gains", "gained", "rise", "rises", "rose", "rally", "rallies",
    "rallied", "surge", "surges", "surged", "soar", "soars", "soared", "jump",
    "jumps", "jumped", "climb", "climbs", "climbed", "advance", "advances",
    "advanced", "rebound", "rebounds", "rebounded", "recover", "recovers",
    "recovered", "recovery", "strong", "stronger", "strength", "robust",
    "solid", "outperform", "outperforms", "outperformed", "beat", "beats",
    "exceed", "exceeds", "exceeded", "upbeat", "optimistic", "bullish",
    "positive", "upgrade", "upgrades", "upgraded", "raise", "raises", "raised",
    "boost", "boosts", "boosted", "record", "high", "highs", "profit",
    "profits", "profitable", "growth", "grow", "grows", "growing", "expand",
    "expands", "expanded", "expansion", "demand", "tailwind", "tailwinds",
    "breakthrough", "approval", "approved", "win", "wins", "won", "award",
    "awarded", "dividend", "buyback", "buybacks", "momentum", "accelerate",
    "accelerates", "accelerating", "milestone", "launch", "launches",
}

# Words that flip the polarity of the next sentiment term.
NEGATORS = {"no", "not", "never", "without", "fails", "fail", "failed",
            "failing", "lack", "lacks", "lacking", "avoid", "avoids",
            "less", "least", "neither", "nor", "cannot", "cant", "can't"}

# High-attention event words: not directional, but signal elevated risk.
EVENT = {"earnings", "guidance", "merger", "acquisition", "acquire", "buyout",
         "ipo", "fda", "ruling", "verdict", "fed", "rate", "cpi", "results",
         "outlook", "forecast", "dividend", "split", "restructuring"}
