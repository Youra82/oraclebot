import pytest

from oraclebot.utils.margin_safety import BITGET_MMR_PCT, SAFETY_BUFFER, compute_max_safe_leverage


def test_wider_barrier_pct_requires_a_lower_max_safe_leverage():
    # Ein weiterer SL-Abstand verlangt, dass Bitgets Liquidationsschwelle NOCH weiter entfernt
    # bleibt -- das ergibt eine NIEDRIGERE sichere Hebel-Obergrenze, keine hoehere.
    narrow = compute_max_safe_leverage(barrier_pct=0.5, taker_fee_pct=0.06)
    wide = compute_max_safe_leverage(barrier_pct=2.0, taker_fee_pct=0.06)
    assert wide < narrow


def test_higher_fees_reduce_max_safe_leverage():
    low_fee = compute_max_safe_leverage(barrier_pct=1.0, taker_fee_pct=0.02)
    high_fee = compute_max_safe_leverage(barrier_pct=1.0, taker_fee_pct=0.20)
    assert high_fee < low_fee


def test_liquidation_price_stays_beyond_barrier_pct_at_the_computed_ceiling():
    """Kernversprechen der Funktion: bei der berechneten Obergrenze muss Bitgets eigene (aus der
    offiziellen Formel abgeleitete) Liquidationsschwelle EXTRA (dank SAFETY_BUFFER) ausserhalb
    der konfigurierten SL-Distanz liegen."""
    barrier_pct = 1.0
    fee_pct = 0.06
    max_safe = compute_max_safe_leverage(barrier_pct=barrier_pct, taker_fee_pct=fee_pct)

    mmr = BITGET_MMR_PCT / 100.0
    fee = fee_pct / 100.0
    liquidation_price_drop_pct = (1.0 - (1.0 - 1.0 / max_safe) / (1.0 - mmr - fee)) * 100
    assert liquidation_price_drop_pct > barrier_pct


def test_safety_buffer_is_strictly_between_zero_and_one():
    # Reine Sanity-Pruefung der Modul-Konstante -- 0 waere immer 0, 1 kein Puffer.
    assert 0.0 < SAFETY_BUFFER < 1.0


def test_zero_fee_and_mmr_reduces_to_pure_barrier_based_ceiling():
    # Mit MMR=0 und Gebuehr=0 vereinfacht sich die Formel auf 1/barrier_pct als theoretische
    # Obergrenze (Liquidation bei exakt 100% Margin-Verlust) -- guter Kontrollfall fuer die Formel.
    max_safe = compute_max_safe_leverage(barrier_pct=1.0, taker_fee_pct=0.0, mmr_pct=0.0)
    theoretical_max = (1.0 / 0.01) * SAFETY_BUFFER  # 1/barrier_pct * Puffer
    assert max_safe == pytest.approx(theoretical_max)


def test_realistic_defaults_yield_a_ceiling_well_below_100x():
    # Bei den aktuellen oraclebot-Standardwerten (barrier_pct=1.0, Gebuehr=0.06) darf 100x
    # NICHT als sicher gelten -- das war genau der real beobachtete Liquidations-Vorfall.
    max_safe = compute_max_safe_leverage(barrier_pct=1.0, taker_fee_pct=0.06)
    assert max_safe < 100
