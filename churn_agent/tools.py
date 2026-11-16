"""Tools exposed to the ADK LLM agent."""
from .model import CONTRACTS, PAYMENTS, get_model, load_data, predict


def dataset_summary() -> dict:
    """Summarize synthetic customer churn training data, without exposing records."""
    df = load_data()
    return {
        'status': 'success', 'source': 'synthetic_demo', 'rows': len(df),
        'churn_rate': round(float(df['churn'].mean()), 3),
        'features': [col for col in df.columns if col != 'churn'],
        'contract_distribution': {str(k): int(v) for k, v in df['contract_type'].value_counts().items()},
    }


def evaluate_churn_model() -> dict:
    """Fit TabPFN on training customers and return held-out test metrics."""
    _, _, metrics = get_model()
    return {'status': 'success', 'model': 'TabPFN-3.5-Fast', 'metrics': metrics,
            'disclaimer': 'Synthetic benchmark data; results are not real-world performance.'}


def predict_customer_churn(
    tenure_months: int,
    monthly_charges: float,
    support_tickets: int,
    contract_type: str,
    payment_method: str,
    auto_pay: bool,
) -> dict:
    """Estimate churn risk for one fictional customer.

    Args:
        tenure_months: Months with the company, between 0 and 120.
        monthly_charges: Monthly bill in dollars, between 0 and 1000.
        support_tickets: Number of recent support tickets, between 0 and 50.
        contract_type: One of month_to_month, one_year, two_year.
        payment_method: One of card, bank_transfer, electronic_check.
        auto_pay: Whether automatic payment is enabled.
    """
    if not 0 <= tenure_months <= 120 or not 0 <= monthly_charges <= 1000 or not 0 <= support_tickets <= 50:
        return {'status': 'error', 'message': 'Numeric fields outside supported bounds.'}
    if contract_type not in CONTRACTS or payment_method not in PAYMENTS:
        return {'status': 'error', 'message': f'contract_type must be {CONTRACTS}; payment_method must be {PAYMENTS}.'}
    record = {
        'tenure_months': tenure_months, 'monthly_charges': monthly_charges,
        'support_tickets': support_tickets, 'contract_type': contract_type,
        'payment_method': payment_method, 'auto_pay': int(auto_pay),
    }
    p = predict(record)
    return {
        'status': 'success', 'churn_probability': round(p, 4),
        'risk_band': 'high' if p >= .65 else ('medium' if p >= .35 else 'low'),
        'inputs': record,
        'note': 'Predictive association, not causal attribution. Demo only.',
    }
