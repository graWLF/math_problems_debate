import asyncio  # noqa
from pathlib import Path
from datetime import datetime
from solib.protocols.protocols import *  # noqa
from solib.Experiment import Experiment
from solib.data.loading import PrOntoQA

# Aşama 1: Pipeline doğrulama — PrOntoQA ile küçük deney
questions = PrOntoQA.data(limit=10)

init_exp = Experiment(
    questions=questions,
    agent_models=[
        "gemini/gemini-2.5-flash",   # Expert Debater (güçlü)
    ],
    agent_toolss=[[]],
    judge_models=[
        "gemini/gemini-3.1-flash-lite-preview",   # Weak Judge (zayıf, eski model)
    ],
    protocols=["blind", "debate"],
    num_turnss=[2],
    bon_ns=[1],
    write_path=Path(__file__).parent
    / "results"
    / f"init_exp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
)


asyncio.run(init_exp.experiment(max_configs=None))
