These two small fixtures render context "Paris is in France." and question
"Where is Paris?" as chunk_multi_turn_tool using the pinned bundled templates.
They were generated with the existing cached model tokenizers by calling
build_messages and apply_chat_template directly, independently of render's span recovery.
The arguments are tools=tools, tokenize=False, add_generation_prompt=True and
enable_thinking=False, matching the external prompt comparator.

The tests reuse the existing cached-tokenizer fixtures with local_files_only=True.
Neither fixture generation nor tests import dac or the comparator.
