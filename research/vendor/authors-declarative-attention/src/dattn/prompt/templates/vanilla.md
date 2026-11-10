# Context

{context}

# Question

{question}

# Instructions

Find the answer to the # Question based solely on information in the # Context above. Reason through the context step by step.

## Answer

End your response with the final answer wrapped in `<answer>...</answer>`. The content depends on the question type:

| Question type | Answer format | Example |
|---|---|---|
| Multiple choice | the letter only | `<answer>D</answer>` |
| Cloze (`<mask-N>`) | the missing word(s) | `<answer>Britney</answer>` |
| Short factual | the value | `<answer>March 14, 2024</answer>` |
| Summary | 2–3 sentences | `<answer>Congress passed the BUILD Act in 2018, creating the IDFC.</answer>` |

After `</answer>`, the response is complete.
