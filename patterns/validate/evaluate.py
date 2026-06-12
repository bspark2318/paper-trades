"""The evaluate-once test gate: verdict logic + the spend-before-compute rows.

The gate's contract: a test_evaluations row is INSERTed and COMMITTED before
any test-period computation begins, verdict 'CRASHED'. Only a completed
evaluation overwrites it. Crashing, aborting, or killing the process spends
the evaluation anyway — peeking is paid for in advance.

SURVIVED requires ALL of:
- test trades exist and mean net return per trade > 0
- beats the TOD-matched random baseline at the Bonferroni-corrected alpha
- train/test consistency: test mean >= 0.25 x train mean; Sharpe same sign
Anything else is REJECTED, with every failed criterion listed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from patterns import db as dbm


@dataclass(frozen=True)
class Verdict:
    survived: bool
    reasons: list[str] = field(default_factory=list)   # failed criteria; empty iff survived

    @property
    def label(self) -> str:
        return "SURVIVED" if self.survived else "REJECTED"


def decide(train_metrics: dict, test_metrics: dict, p_random: float | None, a: float) -> Verdict:
    reasons: list[str] = []

    n_trades = test_metrics.get("n_trades", 0)
    if n_trades == 0:
        return Verdict(False, ["no trades in the test period — the rule never fired"])

    mean = test_metrics["mean_net_ret"]
    if not mean > 0:
        reasons.append(f"test mean net/trade not positive ({mean:+.4%})")

    if p_random is None:
        reasons.append("no random-baseline p-value")
    elif not p_random < a:
        reasons.append(f"does not beat random baseline at corrected threshold "
                       f"(p={p_random:.4f} >= alpha={a:.2e})")

    train_mean = train_metrics.get("mean_net_ret")
    if train_mean is not None and train_mean > 0 and mean < 0.25 * train_mean:
        reasons.append(f"train/test inconsistency: test mean {mean:+.4%} "
                       f"< 0.25 x train mean {train_mean:+.4%}")

    tr_sharpe, te_sharpe = train_metrics.get("sharpe"), test_metrics.get("sharpe")
    if (
        tr_sharpe is not None and te_sharpe is not None
        and not (np.isnan(tr_sharpe) or np.isnan(te_sharpe))
        and np.sign(tr_sharpe) != np.sign(te_sharpe)
    ):
        reasons.append(f"Sharpe sign flip: train {tr_sharpe:.2f} vs test {te_sharpe:.2f}")

    return Verdict(survived=not reasons, reasons=reasons)


def spend_evaluation(conn: sqlite3.Connection, config_hash: str, run_id: int) -> int:
    """Insert + COMMIT the counter row BEFORE computation. Returns its id."""
    cur = conn.execute(
        "INSERT INTO test_evaluations (config_hash, run_id, invoked_at) VALUES (?, ?, ?)",
        (config_hash, run_id, dbm.utcnow()),
    )
    conn.commit()
    rowid = cur.lastrowid
    assert rowid is not None
    return rowid


def record_verdict(conn: sqlite3.Connection, eval_id: int, verdict: Verdict, metrics: dict) -> None:
    conn.execute(
        "UPDATE test_evaluations SET verdict = ?, metrics_json = ? WHERE id = ?",
        (verdict.label, json.dumps(metrics), eval_id),
    )
    conn.commit()


def latest_train_metrics(conn: sqlite3.Connection, config_hash: str) -> dict | None:
    row = conn.execute(
        """SELECT metrics_json FROM runs
           WHERE run_type = 'backtest' AND config_hash = ? AND status = 'ok'
                 AND metrics_json IS NOT NULL
           ORDER BY started_at DESC LIMIT 1""",
        (config_hash,),
    ).fetchone()
    return json.loads(row["metrics_json"]) if row else None
