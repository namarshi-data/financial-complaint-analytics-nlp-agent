import os
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a careful financial-services complaint response assistant.
Draft professional, empathetic, policy-grounded responses.
Do not promise refunds, account changes, credit-report corrections, or legal outcomes.
When risk is high, recommend human escalation.
""".strip()


def _template_response(category: str, priority: str, retrieved_policies: Iterable[str]) -> str:
    policy_summary = "\n".join(f"- {p[:220]}" for p in retrieved_policies) or "- No policy retrieved."
    return f"""Thank you for contacting us. I understand your concern regarding your {category.replace('_', ' ')} complaint.

Based on the information provided, this case is currently classified as **{priority} priority** and should be reviewed according to the applicable complaint-handling procedure.

Relevant policy guidance considered:
{policy_summary}

Recommended next step: verify the customer account details, review supporting documentation, confirm the applicable policy, and escalate to a human specialist if the complaint involves fraud, credit reporting harm, legal risk, or potential SLA breach.

Please note: this draft is for analyst review and should not be sent without confirming the facts of the case."""


def generate_response(category: str, priority: str, complaint_text: str, retrieved_policies: list[str]) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not endpoint or not api_key or not deployment or api_key == "replace-me":
        return _template_response(category, priority, retrieved_policies)

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        policy_context = "\n\n".join(retrieved_policies)
        user_prompt = f"""
Complaint category: {category}
Priority: {priority}

Complaint text:
{complaint_text}

Retrieved policy context:
{policy_context}

Draft a concise, professional response for internal analyst review.
""".strip()
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or _template_response(category, priority, retrieved_policies)
    except Exception as exc:  # noqa: BLE001
        return _template_response(category, priority, retrieved_policies) + f"\n\nAzure OpenAI fallback used: {exc}"
