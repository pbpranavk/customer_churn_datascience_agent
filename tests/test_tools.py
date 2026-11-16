from churn_agent import tools
from scripts.generate_data import generate


def test_dataset_is_reproducible():
    a, b = generate(20), generate(20)
    assert a.equals(b)
    assert set(a.churn.unique()).issubset({0, 1})


def test_summary():
    summary = tools.dataset_summary()
    assert summary['rows'] == 600
    assert 0 <= summary['churn_rate'] <= 1


def test_rejects_bad_contract():
    result = tools.predict_customer_churn(12, 60, 2, 'invalid', 'card', False)
    assert result['status'] == 'error'


def test_predict_with_stub(monkeypatch):
    monkeypatch.setattr(tools, 'predict', lambda _: 0.72)
    result = tools.predict_customer_churn(12, 60, 2, 'month_to_month', 'card', False)
    assert result['risk_band'] == 'high'
    assert result['churn_probability'] == 0.72
