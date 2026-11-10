# Vision-language models, zero-shot

General AI models can't replace a small trained model here yet. On the same 180 wafers, Claude Fable 5.2 reached 0.786 macro-F1, Claude Opus 5.5 0.718 and GPT-6 Luna 0.612, while my CNN got 0.916. This is the full write-up behind the vision-language model section of the [README](../README.md#vision-language-models).

## What I tested

A vision-language model reads images and text together. Zero-shot means it gets no training examples, only a prompt describing the task.

I ran three of them on 180 test wafers, 20 per class, the same ones the random forest and CNN were scored on. GPT-6 Luna is OpenAI's low-cost model. Claude Opus 5.5 and Claude Fable 5.2 are Anthropic's frontier models, and Fable is their top model. All three got the exact same three prompts:

- Label names only.
- One rule per label (v1).
- Corrected rules (v2), explained below.

Macro-F1 averages the F1 score of each class equally, so rare defect types count as much as common ones. The 95% confidence intervals come from a lot bootstrap, which resamples whole lots because wafers in the same lot aren't independent.

## Results

| Same 180 wafers | Macro-F1 (95% CI) | Clean wafers called defects | Defect wafers called clean | Cost per 1,000 wafers |
|---|---|---|---|---|
| Luna, label names only | 0.452 (0.385 to 0.508) | 20 of 20 | 0% | $0.60 |
| Luna, one rule per label (v1) | 0.451 (0.389 to 0.501) | 20 of 20 | 0.6% | $0.42 |
| Luna, corrected rules (v2) | 0.612 (0.538 to 0.682) | 3 of 20 | 17% | $0.47 |
| Opus 5.5, label names only | 0.527 (0.456 to 0.583) | 20 of 20 | 0.6% | $10.65 |
| Opus 5.5, one rule per label (v1) | 0.537 (0.467 to 0.586) | 20 of 20 | 0% | $12.94 |
| Opus 5.5, corrected rules (v2) | 0.718 (0.645 to 0.772) | 5 of 20 | 3.8% | $13.95 |
| Fable 5.2, label names only | 0.727 (0.657 to 0.784) | 8 of 20 | 0% | $23.15 |
| Fable 5.2, one rule per label (v1) | 0.628 (0.565 to 0.674) | 20 of 20 | 0% | $28.30 |
| Fable 5.2, corrected rules (v2) | 0.786 (0.723 to 0.835) | 6 of 20 | 0% | $29.30 |
| Random forest | 0.866 (0.810 to 0.917) | 0 of 20 | 7% | n/a |
| Small CNN | 0.916 (0.866 to 0.953) | 0 of 20 | 2.5% | n/a |

## The prompt fix

My first rule prompt defined "none" as "only a few fails", and that was wrong for this data. On training lots a clean wafer has a median of 9.9% failing dies, and Luna read that normal background as a Random pattern.

The v2 prompt says scattered fails are normal and defines Random as dense, using the training-lot fail shares. On a separate validation sample, v2 raised Luna's macro-F1 from 0.410 to 0.563, and only after that did I score it once on test.

I only found the problem by looking at test results, so v2 is reported next to v1 and doesn't replace it. All three prompts were fixed before Opus or Fable saw a single wafer.

## Opus against Luna

Opus beats Luna on every prompt, by 0.075 to 0.106 macro-F1. All three gaps are clear of zero in a paired lot bootstrap, and the prompt fix helped both models by about the same amount.

Opus's weak spot is Loc. It got 2 of 20 and called 10 of them Edge-Loc, a line that's blurry in the dataset's own labels too (see the [label audit](label-audit.md)).

Three of the five clean wafers Opus flagged, it called Edge-Ring. That fits what [training on simulated wafers](synthetic-training.md) found: real clean wafers carry weak edge structure.

## Fable against Opus

Fable beats Opus on every prompt: by 0.200 macro-F1 with label names only, 0.090 with the v1 rules and 0.068 with v2. All three gaps are clear of zero in a paired lot bootstrap.

With label names only, Fable already scores 0.727, a bit above Opus with my best prompt. It left 12 of 20 clean wafers alone with no rules at all, where Luna and Opus called every one of them a defect.

The flawed v1 rule actually hurt Fable. Told that "none" means only a few fails, it called all 20 clean wafers defects, and its macro-F1 fell from 0.727 to 0.628. The corrected v2 rules brought it back up to 0.786.

Fable never passed a defect wafer as clean, on any prompt. Its weak spot is still Loc: 5 of 20 on v2, with 11 called Edge-Loc.

Fable ran at high effort and Opus at medium, each its own API default. Effort sets how much the model thinks before it answers, so some of this gap could come from that setting.

## Against the trained models

Even the best zero-shot run, Fable with v2, is 0.130 macro-F1 behind the CNN (paired interval 0.068 to 0.198). It's also 0.080 behind the forest (0.007 to 0.162), which is clear of zero, but only just. So none of the three vision-language models can replace labeling here.

All three are good at Scratch, though. On v2, Luna and Fable got 20 of 20 and Opus 19 of 20, against 11 of 20 for the forest and 15 of 20 for the CNN.

## Comparison with Cao and Tao

Cao and Tao (arXiv 2610.35080, 2026) is the paper that got me started on this. They ran GPT-4o zero-shot on 9,000 random WM-811K maps with six classes: none and Near-full dropped, and Edge-Loc and Edge-Ring merged into one Edge class.

To compare, I re-scored our runs their way on the 140 wafers in those classes. Macro accuracy averages accuracy across classes, and overall accuracy is weighted to their class mix.

| Six classes, no "none" | Macro accuracy | Overall accuracy at their class mix |
|---|---|---|
| GPT-4o, plain map (their table 4) | 58.7% | 43.7% |
| GPT-4o, largest fail clusters colored in (their best) | 79.1% | 81.3% |
| Luna, corrected rules (v2) | 59.6% | 54.7% |
| Opus 5.5, corrected rules (v2) | 72.1% | 73.9% |
| Fable 5.2, corrected rules (v2) | 79.2% | 81.2% |
| Random forest | 81.2% | 85.4% |
| Small CNN | 87.9% | 92.4% |

Fable with my v2 rules matches their best GPT-4o run, 79.2% against 79.1% macro accuracy, without any coloring of the map. Opus does a lot better than GPT-4o on a plain map, but not as well as their best run.

Their jump from 43.7% to 81.3% comes with a change to the picture: they color in the largest connected clusters of failing dies before the model sees it. That's the obvious next thing to try here.

The comparison is loose, though. Their sample is random with no lot split. Our models could also answer none or Near-full, which count as wrong here, while theirs only had the six classes to pick from.

And on our side it's only 20 wafers per class (40 for Edge). On these numbers both trained models still beat every zero-shot result, theirs included.

## Costs

Costs use each company's published prices and count every token the calls actually used. Luna ran through the codex CLI, which adds about 15k input tokens to every call.

Opus ran through the Claude Code CLI with its system prompt swapped for one line, so it used about 1.4k input tokens per wafer. But the CLI writes them to a one-hour cache that's billed at twice the input price. Sent with no caching at all, Opus v2 would be $8.23 per 1,000 wafers.

Fable ran the same way. It costs twice as much per token as Opus, so v2 comes to $29.30 per 1,000 wafers as run and $17.82 with no caching.

All three actually ran on subscriptions (ChatGPT credits and Claude Pro), so none of it was billed. Every Claude call is checked to have used my Claude sign-in and no API key, and a call that didn't would be thrown out.

Sources: `results/vlm_summary.csv`, `results/vlm_pairs.csv`, `results/vlm_paper_style.csv`, d46 to d48, d53, d55, d61, d65, d66, d68
