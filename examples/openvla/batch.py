import numpy as np
import sglang as sgl
from .token2action import TokenToAction, image_qa

converter = TokenToAction()

def batch():
    arguments = [
        {
            "image_path": "images/robot.jpg",
            "question": "In: What action should the robot take to {<INSTRUCTION>}?\nOut:",
        }
    ] * 100
    states = image_qa.run_batch(
        arguments,
        max_new_tokens=7,
        temperature=0,
    )
    for s in states:
        print(s.get_meta_info("action")["output_ids"])


if __name__ == "__main__":
    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
    )
    sgl.set_default_backend(runtime)

    batch()

    runtime.shutdown()
