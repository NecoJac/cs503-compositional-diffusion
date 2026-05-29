"""Expose the nested proposal_methods package when running from the repo root."""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent / "proposal_methods"))
