import os

from PyPDF2 import PdfReader

import anthropic


os.environ["ANTHROPIC_API_KEY"] = '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'


def main(in_path):
    anthropic_client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
    reader = PdfReader(in_path)
    text = "\n".join([page.extract_text() for page in reader.pages])

    no_tokens = anthropic.count_tokens(text)
    print(f"Number of tokens in text: {no_tokens}")

    if no_tokens > 100000:
        raise ValueError(f"Text is too long {no_tokens}.")

    prompt = f"{anthropic.HUMAN_PROMPT}: Following text is a student to apply to a university." \
             f"What college or school and majors or academic interests to which the student is applying to? " \
             f"Just return the result in a json dictionary, the keys of the dictionary are 'college', 'first major', " \
             f"'second major', 'third major'. If do not have answer for the key, just return empty string for that key. " \
             f"\n\n{text}\n\n{anthropic.AI_PROMPT}:\n\n"
    res = anthropic_client.completion(prompt=prompt, model="claude-v1.3-100k", max_tokens_to_sample=1000)
    print(res["completion"])

    return res["completion"]


if __name__ == "__main__":
    main('/Users/chengyu/stoooges/gpt_test/data/侯佳盈_University of California-Los Angeles_2022-23_912862.pdf')