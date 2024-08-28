"""
Usage: python3 srt_example_llava.py
"""

import multiprocessing
import time

from tqdm import tqdm

import sglang as sgl


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path) + question
    s += sgl.gen("action")


def single(i):
    t0 = time.time()
    state = image_qa.run(
        image_path="images/robot.jpg",
        question="In: What action should the robot take to {<INSTRUCTION>}?\nOut:",
        max_new_tokens=7,
        temperature=0,
    )
    output_ids = state.get_meta_info("action")["output_ids"]
    print(f"Processs {i} in {time.time()-t0} at {time.time()} ")


if __name__ == "__main__":
    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
    )
    sgl.set_default_backend(runtime)
    for i in tqdm(range(20)):
        output_ids = single(0)
    for i in tqdm(range(50)):
        process = multiprocessing.Process(target=single, args=(i,))
        process.start()
        time.sleep(0.1)
    process.join()
    print("Complete")

    runtime.shutdown()
