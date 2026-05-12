from PoissonPromptGenerator import PoissonPromptGenerator
from queue import Queue

g = PoissonPromptGenerator(
    arrival_rate=1,
    prompt_queue=Queue(),
    max_queue_size=10,
    dataset_name="cais/mmlu",
    dataset_config="all",
    dataset_split="validation",
    prompt_style="mmlu",
    qa_include_context=False,
    force_final_tag=True,
    final_tag="final",
)

x = g.get_next_prompt()
print(x["prompt"])
print("gold:", x["output"])