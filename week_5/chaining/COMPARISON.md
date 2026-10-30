# Chaining vs. Single Call: Comparison Notes

Both scripts in `start/` were run against the same customer question (the frustrated
password-reset message) with `gpt-4o-mini-2024-07-18`. Both scripts were instrumented to
report API calls, token usage, and wall-clock time so the comparison is based on
measurements rather than intuition.

## Measured results (one run each)

| Metric | Chainless (1 call) | Chained (4 calls) |
|---|---|---|
| API calls | 1 | 4 |
| Prompt tokens | 194 | 725 |
| Completion tokens | 313 | 670 |
| Total tokens | 507 | 1,395 |
| Wall-clock time | 5.5 s | 10.4 s |
| Slowest step | n/a | strategy (3.9 s) |

The chained version used about 2.75x the tokens and took about 1.9x as long. Prompt tokens
grew the most because the original question and every intermediate output are re-sent on
later steps.

## What the outputs looked like

**Chainless.** One JSON object with `title`, `analysis`, `strategy`, and `response`. The
model wrapped it in a markdown code fence, so the script had to strip the fence before
`json.loads` would succeed. The analysis fields were terse one-liners. The response was
solid but generic.

**Chained.** The title and analysis steps were clearly better than their chainless
counterparts: the analysis came back as structured JSON with nested detail (three
distinct challenges, three pieces of information to collect). The final response was more
specific, with a concrete checklist of email-format issues.

The chain also exposed a real defect that the single call hid. The **strategy step did not
produce a strategy**. Given only the analysis JSON and the instruction "create a response
strategy," the model wrote a complete customer-facing email. The final step then received
that email as its "strategy" and essentially rewrote it. The pipeline still produced a
good answer, but one of the four steps was doing the wrong job. In the chainless run the
equivalent failure would be invisible inside a single blob of output. This is the
strongest argument for chaining in practice: each link is observable and testable on its
own.

## Answers to the README questions

### Advantages of breaking the task into smaller steps

- **Observability.** Every intermediate result is printed and stored. The strategy-step
  defect above was obvious in seconds.
- **Targeted fixes.** The strategy prompt can be tightened (for example, "Return 3 to 5
  bullet points. Do not write the customer response.") without touching the other three
  prompts or re-validating the whole system.
- **Reliable structure.** Each step asks for one thing, so the JSON-only step reliably
  returns JSON and the title step reliably returns five words. The single prompt mixes
  free text and JSON in one output and already needed fence-stripping to parse.
- **Reuse.** The title and analysis can feed other systems (ticket routing, sentiment
  dashboards) without regenerating the customer response.
- **Model mixing.** Cheap steps like the title can run on a smaller model while the final
  response uses a stronger one. A single call forces one model for everything.

### When a single call is preferable

- The task is genuinely one task, not four. If the product only needs the customer
  response, the title, analysis, and strategy are overhead.
- Latency matters more than inspectability, such as a live chat widget where 5 seconds
  versus 10 seconds is noticeable.
- Cost is tight and volume is high. At roughly 2.75x the tokens, chaining adds up quickly.
- Prototyping. A single prompt is the fastest way to discover whether the model can do the
  job at all before investing in a pipeline.

### Tradeoffs

| Dimension | Single call | Chain |
|---|---|---|
| API cost | Lowest. 507 tokens here. | About 2.75x higher because context is re-sent per step. |
| Response time | Fastest. One round trip. | Slower. Four sequential round trips. Parallelizing independent steps (title and analysis) would recover some time. |
| Maintainability | One large prompt that is fragile to edit. Changing one instruction can shift every part of the output. | Four small prompts with clear contracts. Each can change independently. |
| Debugging | Hard. A bad response gives no clue which "mental step" went wrong. | Easy. Intermediate outputs point directly at the failing step. |
| Output quality | Adequate but generic. Analysis is shallow because the model splits attention across four goals. | Richer analysis and a more specific response. Also surfaces a mis-behaving step so quality can be improved deliberately. |

## Follow-up improvements worth making

- Rewrite the strategy prompt so it returns bullet points, not prose addressed to the
  customer, and assert on the output shape.
- Run the title and analysis steps concurrently. They only depend on the question.
- Add `response_format={"type": "json_object"}` to the analysis step and to the chainless
  prompt so neither needs fence-stripping.
- Trace the chain in LangSmith (week 4's `@traceable`) so each step's tokens and latency
  are recorded automatically instead of by hand.

---
Generated by Rocket Flow · 2.0.20 · 2026-10-29
