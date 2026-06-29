# Roadmap

Marshmello’s roadmap focuses on educational model quality, honest evaluation, and clear reproducibility.

## Near Term

- Finish 300M smoke testing.
- Validate memory, throughput, checkpoint size, and resume behavior for `large_300m`.
- Run initial 18J and 18K evaluations only after a real 300M training pass.
- Improve evaluation reports so each checkpoint has a clear promotion or rollback decision.

## Training

- Chat-only pretraining for 300M.
- Teacher SFT v2 with stricter regression gates.
- Broad assistant SFT after the 300M baseline is stable.
- Keep short, low-learning-rate SFT runs as the default until benchmarks justify longer runs.

## Data And Tokenization

- Arabic support.
- Multilingual tokenizer experiments.
- Better coverage for punctuation, chat markers, and non-English scripts.
- Clear dataset cards for each publishable dataset variant.

## Evaluation

- Better evaluation reports.
- Cleaner 18J / 18K checkpoint comparisons.
- More explicit hallucination and repetition tracking.
- Possible additional held-out sets for multilingual and Arabic behavior.

## Research Options

- Possible long-context support.
- Alternative context lengths after the 300M model is stable.
- More systematic comparison of chat-only adaptation, teacher SFT, and broad SFT order.

## Release Discipline

- Keep benchmark numbers honest.
- Do not publish smoke-test checkpoints as stable models.
- Separate internal benchmark results from public leaderboard claims.
- Keep documentation clear about limitations and intended educational use.
