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
        temperature=0,
    )
    output_ids = state.get_meta_info("action")["output_ids"]
    return output_ids


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
    ouput_ids = single()
    print(ouput_ids)

    batch()

    runtime.shutdown()
