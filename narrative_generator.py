"""
Compliance Narrative Generator
Simple functions for generating compliance narratives with Gemini
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Gemini LLM import
import google.generativeai as genai

# Global LLM state
_llm_model = None


def load_llm(model_name='gemini-2.5-flash', api_key=None):
    """Configure Gemini client for narrative generation."""
    global _llm_model
    
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  No API key. Set GOOGLE_API_KEY or pass api_key parameter")
        return
    
    genai.configure(api_key=api_key)
    _llm_model = model_name
    print(f"✓ Gemini configured for narratives: {model_name}")


def generate_fallback_narrative(transaction_amount: float,
                                client_profile: str,
                                flagged_rule: str,
                                additional_context: Optional[str]) -> str:
    """Generate template-based narrative when LLM is not available."""
    return f"""COMPLIANCE INCIDENT REPORT

Transaction Summary:
A transaction in the amount of ${transaction_amount:,.2f} has been flagged for review under the following compliance rule: {flagged_rule}.

Client Profile:
{client_profile}

Flagged Rule:
{flagged_rule}

Analysis:
This transaction was flagged due to compliance concerns related to {flagged_rule}. The transaction amount of ${transaction_amount:,.2f} exceeds the threshold established in our compliance framework. The client profile indicates {client_profile.lower()}.

{additional_context if additional_context else "No additional context provided."}

Recommended Actions:
1. Review the transaction details and client profile against the compliance rule requirements
2. Conduct enhanced due diligence if required
3. Document the review process and decision
4. Escalate to compliance officer if necessary

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""


def generate_narrative(transaction_amount: float,
                      client_profile: str,
                      flagged_rule: str,
                      additional_context: Optional[str] = None) -> Dict[str, Any]:
    """Generate compliance incident narrative."""
    if _llm_model is None:
        return {
            'narrative': generate_fallback_narrative(transaction_amount, client_profile, flagged_rule, additional_context),
            'transaction_amount': transaction_amount,
            'client_profile': client_profile,
            'flagged_rule': flagged_rule,
            'generated_at': datetime.now().isoformat(),
            'additional_context': additional_context
        }
    
    prompt = f"""You are a compliance officer writing an incident report. Generate a clear, professional, and auditable narrative explaining a flagged transaction.

Transaction Details:
- Amount: ${transaction_amount:,.2f}
- Client Profile: {client_profile}
- Flagged Rule: {flagged_rule}
{f'Additional Context: {additional_context}' if additional_context else ''}

Requirements:
1. Write in a formal, professional tone suitable for internal audit
2. Clearly explain why the transaction was flagged
3. Reference the specific compliance rule
4. Describe the client profile and transaction context
5. Include recommended next steps or actions
6. Be factual and objective - avoid speculation
7. Keep the narrative concise but complete (200-400 words)

Generate the narrative:"""

    narrative = None
    try:
        model = genai.GenerativeModel(_llm_model)
        response = model.generate_content(prompt)
        candidates = getattr(response, "candidates", [])
        if candidates and candidates[0].content.parts:
            narrative = candidates[0].content.parts[0].text
        else:
            narrative = response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        narrative = f"Error generating narrative: {e}"
    
    return {
        'narrative': narrative,
        'transaction_amount': transaction_amount,
        'client_profile': client_profile,
        'flagged_rule': flagged_rule,
        'generated_at': datetime.now().isoformat(),
        'additional_context': additional_context
    }


def main():
    """Example usage."""
    print("="*80)
    print("Compliance Narrative Generator")
    print("="*80)
    
    # Load LLM
    load_llm()
    
    # Generate narrative
    result = generate_narrative(
        transaction_amount=75000.00,
        client_profile="New individual customer, high-risk jurisdiction, cash deposit",
        flagged_rule="Section 13(e) - Transactions above $50,000 require enhanced due diligence",
        additional_context="Customer declined to provide additional documentation."
    )
    
    print("\nGenerated Narrative:")
    print("="*80)
    print(result['narrative'])


if __name__ == "__main__":
    main()
