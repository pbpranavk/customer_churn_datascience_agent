"""Generate reproducible, entirely synthetic customer records."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(25, 125, n), 2)
    tickets = np.clip(rng.poisson(1.3, n), 0, 8)
    contract = rng.choice(['month_to_month', 'one_year', 'two_year'], n, p=[.55, .28, .17])
    payment = rng.choice(['card', 'bank_transfer', 'electronic_check'], n, p=[.48, .30, .22])
    autopay = rng.binomial(1, .58, n)
    score = (-1.2 + 1.20 * (contract == 'month_to_month')
             - .032 * tenure + .012 * (monthly - 65)
             + .38 * tickets - .55 * autopay + .38 * (payment == 'electronic_check'))
    p = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, p)
    return pd.DataFrame({
        'tenure_months': tenure, 'monthly_charges': monthly,
        'support_tickets': tickets, 'contract_type': contract,
        'payment_method': payment, 'auto_pay': autopay, 'churn': churn,
    })


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'churn_agent' / 'data' / 'customers.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(out, index=False)
    print(f'Wrote {out}')
