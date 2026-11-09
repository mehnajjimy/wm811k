# Grounded triage assistant

When an AI model drafting a fab incident report is limited to one row of my root-cause table, it stops making things up. With the row, 60 of 60 drafts passed every check, and without it only 2 of 60 did.

This is the full write-up behind the triage section of the [README](../README.md#an-ai-that-drafts-incident-reports-without-making-things-up).

## What it does

When the CNN flags a wafer, GPT-6 Luna drafts the first part of an 8D response. 8D is a standard problem-solving format in manufacturing. Its D2 to D4 steps describe the problem, contain it and look for the root cause.

Luna gets the CNN's prediction, the wafer's die yield, the product's clean-wafer yield and the lot context. It's told to use only the matching row of the [root-cause table](root_causes.md). Grounding means tying the model to a fixed source like this, so it can't pull in claims from elsewhere.

## How I checked it

Code checks every reply, with no AI judge involved. It looks for:

- process modules the table row doesn't support
- sources outside the row's tags
- numbers that aren't in the facts it was given
- named tools
- the wrong pattern

I ran it on 54 test wafers plus 6 trick questions, once with the table row and once without it as a control.

## Results

| 54 test wafers + 6 trick questions | All checks pass | Module beyond the table | Outside reference cited | Number not in the facts |
|---|---|---|---|---|
| With the table row | 60 of 60 | 0% | 0% | 0% |
| Without the table (control) | 2 of 60 | 97% | 22% | 7% |

The trick questions ask for a tool ID, a dollar cost, a journal reference for a row that has none, and how many other wafers are affected. With the table, five of the six listed the requested item as unknown.

The sixth asked whether implant caused an Edge-Ring. It never named implant and listed the causal step as unknown.

Without the table, replies often add lithography, implant or thermal causes for a Center wafer. Some cite lecture or thesis URLs I haven't checked.

## What it doesn't show

Grounding keeps the model inside what the project can back up. That doesn't mean the table is right, since most rows are still hypotheses.

And the checks only catch invented specifics. A weak argument made only of allowed words gets through.

Sources: `results/assistant_summary.csv`, d49, d50, d60, d62
