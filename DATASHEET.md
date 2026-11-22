# Datasheet for PDC-Bench (abridged)

**Motivation.** Evaluate compliance with semantic privacy constraints on the disclosure
granularity of in-context personal attributes, and measure semantic leakage continuously
over a KG hierarchy.

**Composition.** Four domains (geography, age, race, occupation); 323 KG nodes following
hierarchies designed for this benchmark; the top level of the race and ethnicity KG
reproduces the seven minimum categories of OMB Statistical Policy Directive No. 15. One synthetic profile per KG node, generated with GPT-4o. All
profiles are fictional; any resemblance of generated names to real persons is coincidental.
No real personal data is contained in this benchmark.

**Sensitive content.** The age and race domains include discrimination-sensitive task
texts (hiring, credit, housing, education, health/insurance) that instantiate legally
restricted decision contexts. They exist solely to evaluate whether models condition
outputs on protected attributes; they must not be used to build systems that do so.
Outputs of these tasks were excluded from human annotation for annotator welfare; they
are machine-judged by the identical procedure as other tasks.

**Collection.** Profiles and model outputs are model-generated; tasks and constraints were
authored by the researchers. Profiles were generated with GPT-4o, and outputs were generated
by the eleven evaluated models. Human annotation of 400 outputs was performed by eight annotators
following the guideline in `annotation/`, on non-discrimination outputs only. The
annotators are graduate students or researchers at a university in Japan; one of them is an
author of the paper, and the other seven were paid 2,000 JPY per hour and took part
voluntarily.

**Uses.** Benchmarking privacy-constraint compliance of LLMs. Not a source of facts about
real people; not for training disclosure systems.

**Distribution.** Code under the MIT License, data under CC BY 4.0; copyright held by
the authors.
