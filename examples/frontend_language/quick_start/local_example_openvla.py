"""
Usage: python3 local_example_openvla.py
"""

import numpy as np
from transformers import AutoConfig

import sglang as sgl


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path) + question
    s += sgl.gen("action")


class TokenToAction:
    def __init__(self, n_action_bins: int = 256, unnorm_key: str = "bridge_orig"):
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = 32000
        self.unnorm_key = unnorm_key
        self.config = AutoConfig.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        ).to_dict()
        self.norm_stats = self.config["norm_stats"]
        assert unnorm_key is not None
        if unnorm_key not in self.norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {self.norm_stats.keys()}"
            )

    def convert(self, output_ids):
        predicted_action_token_ids = np.array(output_ids)
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions


converter = TokenToAction()


def single():
    state = image_qa.run(
        image_path="images/robot.jpg",
        question="In: What action should the robot take to {<INSTRUCTION>}?\nOut:",
        max_new_tokens=7,
        temperature=0,
    )
    output_ids = state.get_meta_info("action")["output_ids"]
    print(output_ids)
    assert output_ids == [[31888,31869,31900,31912,31823,31882,31744]]
    action = converter.convert(output_ids)
    assert np.array_equal(np.round(action, 5), np.round([[-3.78757518e-03,5.47156949e-04,-2.41243806e-04,-2.50440557e-02,2.53441257e-02,-1.77964902e-02,9.96078431e-01]], 5))
    return action

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

    # batch()

    runtime.shutdown()