# <Decision-relevant finding> — <observation ID>

> **Status:** <PLANNED | PILOT | REPLICATED | QUALIFIED | BLOCKED> · **Evidence:** <REPORTED | VERIFIED | INDEPENDENTLY_AUDITED> · **As of:** <YYYY-MM-DD> · **Run(s):** <immutable IDs>

**Bottom line.** <One evidence-backed sentence answering the research question, with baseline, comparison, units, and scope. State "no result yet" if unmeasured.>

**Decision / next gate.** <Adopt, reject, continue testing, or defer; name one concrete next action and the condition that could change the decision.>

## Evidence at a glance

| Decision metric (unit; statistic) | Baseline | Candidate | Change | Coverage |
| --- | ---: | ---: | ---: | --- |
| <Qualified task quality / scoring rule> | <value> | <value> | <pp or absolute> | <n/N> |
| <E2E latency, p50 or p95, s> | <value> | <value> | <speedup ×> | <paired n, CCU> |
| <Aggregate output throughput, tok/s> | <value> | <value> | <ratio ×> | <CCU, load> |
| <Peak device VRAM, GiB> | <value> | <value> | <difference GiB> | <measurement window> |
| <EAGLE3 accepted/drafted, % (if relevant)> | N/A | <value> | N/A | <accepted/drafted> |

*Retain only decision-relevant rows. Mark missing measurements `NOT_MEASURED`, not zero; use `N/A` only when inapplicable. Identify mean vs median, CI, and ratio definition when they change the conclusion.*

**Tested scope.** <Model/draft exact revisions; dataset/release and original-vs-reduced context; N and selection; GPU; serving stack; decoding policy; CCU and repetition/arrival pattern. Keep to 1–2 lines.>

**Decision-critical caveat.** <Most important limitation, validity threat, or blocker; distinguish official publisher-scored accuracy from local proxy judgments; state whether the result generalizes.>

**Evidence / reproduction.** [run manifest](<path-or-URL>) · [raw per-request results](<path-or-URL>) · [config and commands](<path-or-URL>) · [independent audit](<path-or-URL-or-NOT_AUDITED>) · [living observation plan](<path-or-URL>)

<!-- Publish as a concise manager-facing report. Replace every placeholder, remove any irrelevant row, and link all claims to immutable evidence. Keep extended settings, CCU×context grids, plots, per-sample scoring and limitations in separate technical artifacts, not in this executive page. -->
