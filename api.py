from typing import Optional
import logging

import fire
from flask import Flask, request
from llama import Llama
import torch.distributed as dist


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    port: int = 8000,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    if dist.get_rank() == 0:
        app = Flask(__name__)

        @app.route("/api/complete", methods=["POST"])
        def complete():
            inputs = request.json["inputs"]
            messages = [{"role": "user", "content": inputs}]
            res = generator.chat_completion([messages])
            logging.warning(f'{inputs}: {res}')
            return {
                "generated_text": res[0]["generation"]["content"].strip(),
                "status": 200,
            }

        app.run(host='0.0.0.0', port=port)


# codellama/CodeLlama-13b-hf
if __name__ == "__main__":
    fire.Fire(main)

