# Backend findings retained from the September 2026 probes

## 0.1 Plugin allow-list

An empty VLLM_PLUGINS is an explicit allow-list selecting no plugins. This bypassed the unfinished plugin during the initial probes. Production DA must include dattn or leave this variable unset so CUSTOM registration runs.

## 0.2 Greedy logits processors

On vLLM 0.29.0, argmax-invariant processors are applied only on the random-sampling branch. The initial greedy probe constructed the processor and called update_state, but never apply. DattnLogitsProcessor must return False from is_argmax_invariant, and its FQCN must be passed through logits_processors so output token IDs are retained.

## 1 Model runner and FSM lag

The V1 runner booted both Gemma-4-31B-it and Qwen3.6-27B. Both synchronous and asynchronous probes observed one-token FSM lag. The output list is a live reference, with its last valid ID belonging to the previous step. Negative IDs must not advance the FSM. Qwen uses three Mamba groups and one full-attention group, whose native planner needs the compacted CPU lengths when TRT-LLM attention is disabled.

## 2.0 CUDA graph measurements

Dummy-run probes appeared insensitive to compacted lengths because they did not exercise real scheduled decode and graph replay. Subsequent execute_model measurements showed that FULL replay uses the staged scratch metadata and responds to shrinkage. Record the actual per-step replay mode, retain scratch pointer identity and test tail poisoning against staged corruption. Forward CUDA events and scheduler-to-scheduler wall periods measure different scopes.

## 3 Historical parity findings

These findings explain differences from the private dac runs. They are evidence about
those runs, not runtime waivers in the public mask/prompt self-check. The original
comparator and its tests are recoverable at 7c97929:src/dattn/eval/parity.py and
tests/unit/eval/test_parity.py.

- **gemma_first_segment_span_starts_3_tokens_earlier**: the public Gemma span starts
  three tokens earlier than dac, retaining the model turn header with the pair it
  opens. The reference regex started at the tool-call marker. The difference is zero
  after block rounding to 32. Evidence: tests/unit/prompt/render_golden.json
  first_span_lead and DESIGN.md at 7c97929, section 7 (F D1).
- **pad_tokens_counted_in_total_attendable**: the public denominator counts padded
  tokens, matching the paper definition. The reference skipped them. Evidence:
  docs/PROVENANCE.md P5 and the protected FSM ledger tests.
- **contexts_quoting_chatml_tool_markup**: per-family reference regexes could truncate
  spans when context quoted ChatML tool markup. Prefix rendering and body containment
  avoid this. Evidence: docs/PROVENANCE.md P1 and protected prompt tests.
- **xml_ids_out_of_order_or_duplicated**: reference validation sorted IDs, accepting
  out-of-order IDs that it later resolved by dictionary. The public renderer requires
  contiguous ascending IDs. Evidence: docs/PROVENANCE.md P7, dac/base.py:82,127-139.
- **qwen_tool_schema_present**: the public Qwen prompt declares the tool. The reference
  Qwen template omitted it. This is a symptom of the bundled-template mechanism below.
- **dac_served_bundled_assistant_mask_chat_template**: dac serve.py:1117-1120 replaced
  the Hub template through maybe_apply_assistant_mask_chat_template. The registry was
  dac/src/dac/templates/chat_templates.py:70-79. On all 1611 headline rows, stock minus
  bundled was +2 tokens for Gemma and +315 for Qwen. Gemma's bundled template omitted
  the 8-character closing tool-block marker. Qwen's template never referenced tools,
  dropping the 1361-character tool declaration. Recorded prompt counts matched the
  bundled counts on all rows. With the same template, both renderers produced identical
  prompt text on all 1611 rows for both DA arms and both models. Evidence:
  an internal prompt-parity report, recorded September 17, 2026.
  The reference length filter nevertheless used the stock tokenizer
  (run_eval.py:1234), shifting Qwen's admission counts by 315 tokens.
  One separate Qwen row, longbench_v2_multidoc_qa#108, had identical rendered text but
  an 18-token difference between tokenizers 0.23.2 and 0.22.2 at combining-solidus
  followed by =j or =i sites. That difference is tokenizer-version evidence,
  not a template difference.
