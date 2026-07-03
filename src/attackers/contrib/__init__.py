"""CLEAVE attacker registration.

The framework auto-discovers attackers under ``src/attackers/contrib`` at import
time (see ``src.attackers._load_contrib_attackers``). We expose an ``ATTACKERS``
mapping and register it here so the ``cleave`` attacker is available to the runner
and to the Hydra configs (``conf/attacker/cleave.yaml``).
"""
from src.attackers import register_attacker
from src.attackers.contrib.cleave_attack import CLEAVEAttack

ATTACKERS = {"cleave": CLEAVEAttack}

for _name, _cls in ATTACKERS.items():
    register_attacker(_name, _cls)
