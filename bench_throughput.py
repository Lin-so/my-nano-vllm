from nanovllm import LLM


def main():
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    llm = LLM("/home/models/Qwen3-4B-FP8/")

    llm.bench(num_seqs,max_input_len,max_ouput_len)


if __name__ == "__main__":
    main()
