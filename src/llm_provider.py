# src/llm_provider.py

import os
import yaml
from dotenv import load_dotenv
from litellm import completion


class LLMProvider:
    """
    Handles:
    1. LiteLLM connection
    2. Gemini API through LiteLLM
    3. Final grounded answer generation
    4. Prompts loaded from config/prompts.yaml
    """

    def __init__(
        self,
        model_name="gemini/gemini-2.5-flash-lite"
    ):
        load_dotenv()

        self.model_name = model_name

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in your .env file."
            )

        # LiteLLM uses this internally
        os.environ["GEMINI_API_KEY"] = api_key

        self.prompts = self.load_prompts()

    def load_prompts(self):
        """
        Load prompts from config/prompts.yaml
        """
        with open(
            "config/prompts.yaml",
            "r",
            encoding="utf-8"
        ) as file:
            return yaml.safe_load(file)

    def generate_answer(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate final answer using ONLY provided context
        with prompt versioning support.
        """

        prompt_template = self.prompts["answer_generation_prompt"]

        final_prompt = prompt_template.format(
            query=query,
            context=context
        )

        try:
            response = completion(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
                temperature=0.1
            )

            final_answer = response["choices"][0]["message"]["content"]

            return final_answer.strip()

        except Exception as e:
            print(f"LLM Error: {str(e)}")

            return (
                "I cannot generate a safe medical answer at this time."
            )


if __name__ == "__main__":
    """
    Quick test:
    python src/llm_provider.py
    """

    provider = LLMProvider()

    sample_query = "When should insulin therapy begin for diabetes?"

    sample_context = """
Source: diabetes_guidelines.pdf
Page: 42

Patients with severe hyperglycemia should be considered
for early insulin therapy initiation.

Source: diabetes_guidelines.pdf
Page: 45

Insulin may also be initiated when oral therapy fails
to achieve glycemic control.
"""

    answer = provider.generate_answer(
        query=sample_query,
        context=sample_context
    )

    print("\nGenerated Answer:\n")
    print(answer)