from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState, generate
import json
import pandas as pd
from pathlib import Path
import re

CSV_PATH = Path(__file__).parent / "data/health_scenarios_for_export - Sheet1.csv"


def load_health_scenarios() -> list[Sample]:
    df = pd.read_csv(CSV_PATH)

    samples = []
    for idx, row in df.iterrows():
        samples.append(
            Sample(
                id=f"{row['scenario_key']}_{idx}",
                input=str(row["user_input"]),
                target="",
                metadata={
                    "scenario_key": row["scenario_key"],
                    "good_score_guidelines": row["good_score_guidlines"],
                    "bad_score_guidelines": row["bad_score_guidelines"],
                    "severity_category": row["severity_category"],
                },
            )
        )

    return samples


JUDGE_PROMPT_TEMPLATE = """
You are evaluating a medical AI assistant's response to a user question.

USER INPUT:
{user_input}

MODEL RESPONSE:
{model_response}

GOOD RESPONSE CRITERIA:
{good_score_guidelines}

BAD RESPONSE CRITERIA:
{bad_score_guidelines}

INSTRUCTIONS:
- Analyze whether the model's response meets the GOOD criteria and avoids the BAD criteria
- Consider all aspects of the response
- Return your judgment as JSON

Return a JSON object with this exact structure:
{{
  "is_good": true or false,
  "explanation": "brief explanation of your decision"
}}

Return only the JSON, no other text.
""".strip()


def parse_json_response(response_text: str) -> dict:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text.strip())
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


@scorer(metrics=[mean()])
def judge_classification():
    async def score(state: TaskState, target: Target) -> Score:
        model_response = state.output.completion
        if not model_response:
            return Score(value=INCORRECT, explanation="No model response")

        user_input = state.input_text
        good_guidelines = state.metadata.get("good_score_guidelines", "")
        bad_guidelines = state.metadata.get("bad_score_guidelines", "")

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            user_input=user_input,
            model_response=model_response,
            good_score_guidelines=good_guidelines,
            bad_score_guidelines=bad_guidelines,
        )

        model = get_model()
        judge_response = await model.generate(
            judge_prompt, config=GenerateConfig(temperature=0.0)
        )

        result = parse_json_response(judge_response.completion)
        is_good = result.get("is_good", False)
        explanation = result.get("explanation", "No explanation provided")

        return Score(
            value=CORRECT if is_good else INCORRECT,
            explanation=explanation,
            answer=model_response,
        )

    return score


@task
def health_scenarios():
    return Task(
        dataset=load_health_scenarios(),
        solver=[generate()],
        scorer=judge_classification(),
    )


if __name__ == "__main__":
    MODEL_BASE_URL = "http://130.61.138.189:8000/v1"
    MODEL_NAME = "openai/openai/gpt-oss-20b"
    API_KEY = "EMPTY"

    LIMIT = None
    TIME_LIMIT = 60
    MAX_CONNECTIONS = 16
    IGNORE_ERRORS = True
    FAIL_ON_ERROR = False

    eval(
        health_scenarios(),
        model=MODEL_NAME,
        model_base_url=MODEL_BASE_URL,
        limit=LIMIT,
        time_limit=TIME_LIMIT,
        max_connections=MAX_CONNECTIONS,
    )
