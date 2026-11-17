"""Contrib attacker package.

The framework auto-discovers attackers under ``src/attackers/contrib`` at import
time (see ``src.attackers._load_contrib_attackers``); each module registers its
own ``ATTACKERS`` mapping (e.g. ``undercut``).
"""
