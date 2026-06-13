"""Live paper-execution layer: the intraday trade loop and status reporting.

Everything here drives the Broker protocol and a bar feed — never alpaca types
directly — so the loop is exercised in tests against a mocked broker + scripted
clock with no network and no sleeps.
"""
