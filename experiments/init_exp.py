import asyncio  # noqa
from pathlib import Path
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import LogiQA

# Pilot deney: 10 soru, blind + debate, Groq free tier limitine sığacak şekilde
questions = LogiQA.data(limit=30, augmented=True)

CONTINUE_FROM = Path(__file__).parent / "results" / "logiqa_groq_2026-05-06_21-40-59"

init_exp = Experiment(
    questions=questions,
    agent_models=[
        "groq/llama-3.3-70b-versatile",   # Expert Debater (70B)
    ],
    agent_toolss=[[]],
    judge_models=[
        "groq/llama-3.1-8b-instant",       # Weak Judge (8B)
    ],
    protocols=["blind", "debate", "consultancy"],
    num_turnss=[2],
    bon_ns=[1],
    write_path=CONTINUE_FROM,
    continue_from=CONTINUE_FROM,
)


asyncio.run(init_exp.experiment(max_configs=None))
