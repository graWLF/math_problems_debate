import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import LogiQA

# Aşama 4: Groq modelleriyle asimetrik deney — LogiQA + sentetik distractorlar
questions = LogiQA.data(limit=50, augmented=True)

init_exp = Experiment(
    questions=questions,
    agent_models=[
        "groq/llama-3.3-70b-versatile",   # Expert Debater (70B — güçlü)
    ],
    agent_toolss=[[]],
    judge_models=[
        "groq/llama-3.1-8b-instant",       # Weak Judge (8B — zayıf)
    ],
    protocols=["blind", "debate", "consultancy"],
    num_turnss=[2, 4],
    bon_ns=[1],
    write_path=Path(__file__).parent
    / "results"
    / f"logiqa_groq_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)


asyncio.run(init_exp.experiment(max_configs=None))
