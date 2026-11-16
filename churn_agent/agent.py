"""ADK entrypoint discovered by `adk web` and `adk deploy agent_engine`."""
import os
from google.adk.agents import Agent
from .tools import dataset_summary, evaluate_churn_model, predict_customer_churn

root_agent = Agent(
    name='customer_churn_agent',
    model=os.getenv('AGENT_MODEL', 'gemini-2.5-flash'),
    description='Customer churn risk analysis using TabPFN and Google ADK.',
    instruction=(
        'You are a customer churn analysis assistant using a SYNTHETIC dataset. '
        'Use dataset_summary for data questions; evaluate_churn_model for benchmark metrics; '
        'predict_customer_churn for customer-specific churn probability. '
        'Ask for missing required customer fields instead of guessing. '
        'Never invent model scores, evaluation results, explanations, or causal claims. '
        'Explain that risk bands are illustrative thresholds, not calibrated business rules. '
        'Retention suggestions are generic hypotheses, not proven causal interventions. '
        'Do not claim this prototype is production validated.'
    ),
    tools=[dataset_summary, evaluate_churn_model, predict_customer_churn],
)
