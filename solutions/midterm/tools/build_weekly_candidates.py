"""Separate entry point for real weekly-context candidate expansion; no fits."""
from build_weekly_context import main

if __name__=='__main__':
    main(retrieval=True)
