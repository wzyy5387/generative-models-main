# -*- coding: UTF-8 -*-

try:
    from .dad_planner import Planner
except ModuleNotFoundError as exc:
    if exc.name != "gurobipy":
        raise
    # Normalized decision-value utilities do not require the optional solver.
    # Keep Planner unavailable rather than making every package import fail.
    Planner = None
