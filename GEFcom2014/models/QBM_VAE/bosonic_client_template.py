# -*- coding: utf-8 -*-

"""Backward-compatible import for the optional Kaiwu/SPQC client."""

from .kaiwu_adapter import BosonicClient, KaiwuClient

__all__ = ["BosonicClient", "KaiwuClient"]
