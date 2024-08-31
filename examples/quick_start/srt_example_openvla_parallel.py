"""
Usage: python3 srt_example_llava.py
"""

import threading
import time

from tqdm import tqdm

import sglang as sgl


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path) + question
    s += sgl.gen("action")


last_get_time = time.time()


def single(i):
    t0 = time.time()
    state = image_qa.run(
        image_path="images/robot.jpg",
        question="In: What action should the robot take to {<INSTRUCTION>}?\nOut:",
        max_new_tokens=7,
        temperature=0,
    )
    output_ids = state.get_meta_info("action")["output_ids"]
    global last_get_time
    print(
        f"Processs {i} in {time.time()-t0} at {time.time()}, FPS: {1/(time.time()-last_get_time)}"
    )
    last_get_time = time.time()


if __name__ == "__main__":
    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_radix_cache=True,
        # enable_torch_compile=True,
    )
    sgl.set_default_backend(runtime)
    for i in tqdm(range(20)):
        output_ids = single(0)
    threads = []
    for i in range(1000):
        t = threading.Thread(target=single, args=(i,))
        threads.append(t)
        t.start()
        time.sleep(0.05)

    threads[-1].join()
    print("Complete")

    runtime.shutdown()
