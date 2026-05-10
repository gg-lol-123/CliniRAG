# src/llm_provider.py

import os
import yaml

from dotenv import load_dotenv
from litellm import completion


class LLMProvider:
    """
    Handles:
    1. Gemini generation
    2. Ollama fallback generation
    3. Prompt loading
    4. Reliable answer generation
    """

    def __init__(
        self,
        primary_model="gemini/gemini-2.5-flash-lite",
        fallback_model="ollama/llama3.2"
    ):
        load_dotenv()

        self.primary_model = primary_model
        self.fallback_model = fallback_model

        # Gemini API availability
        self.api_key = os.getenv("GEMINI_API_KEY")

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

    def _build_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Build final grounded prompt
        """

        prompt_template = self.prompts[
            "answer_generation_prompt"
        ]

        return prompt_template.format(
            query=query,
            context=context
        )

    def _generate_with_gemini(
        self,
        prompt: str
    ) -> str:
        """
        Generate using Gemini
        """

        print("\nUsing Gemini...\n")

        response = completion(
            model=self.primary_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )

        return response["choices"][0]["message"]["content"]

    def _generate_with_ollama(
        self,
        prompt: str
    ) -> str:
        """
        Generate using Ollama
        """

        print("\nSwitching to Ollama fallback...\n")

        response = completion(
            model=self.fallback_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )

        return response["choices"][0]["message"]["content"]

    def generate_answer(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate grounded answer
        with Gemini -> Ollama fallback.
        """

        final_prompt = self._build_prompt(
            query=query,
            context=context
        )

        # -----------------------------------
        # Try Gemini first
        # -----------------------------------

        if self.api_key:

            try:

                answer = self._generate_with_gemini(
                    final_prompt
                )

                return answer.strip()

            except Exception as gemini_error:

                print("\nGemini failed.")
                print(
                    f"Reason: {str(gemini_error)}"
                )

        # -----------------------------------
        # Ollama fallback
        # -----------------------------------

        try:

            answer = self._generate_with_ollama(
                final_prompt
            )

            return answer.strip()

        except Exception as ollama_error:

            print("\nOllama fallback failed.")
            print(
                f"Reason: {str(ollama_error)}"
            )

            return (
                "I cannot generate a safe medical answer at this time."
            )


if __name__ == "__main__":

    provider = LLMProvider()

    sample_query = (
        "When should insulin therapy begin for diabetes?"
    )

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