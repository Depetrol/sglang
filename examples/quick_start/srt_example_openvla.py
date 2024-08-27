"""
Usage: python3 srt_example_llava.py
"""

import sglang as sgl


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path) + question
    s += sgl.gen("action")


def single():
    state = image_qa.run(
        image_path="images/robot.jpg",
        question="In: What action should the robot take to {<INSTRUCTION>}?\nOut:",
        max_new_tokens=7,
    )
    output_ids = state.get_meta_info("action")["output_ids"]
    return output_ids


def batch():
    states = image_qa.run_batch(
        [
            {"image_path": "images/robot.jpg", "question": "What is this?"},
            {"image_path": "images/dog.jpeg", "question": "What is this?"},
        ],
        max_new_tokens=128,
    )
    for s in states:
        print(s["answer"], "\n")


if __name__ == "__main__":
    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_cuda_graph=True,
        disable_radix_cache=True,
    )
    sgl.set_default_backend(runtime)
    ouput_ids = single()
    print(ouput_ids)
    runtime.shutdown()
