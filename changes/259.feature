Added ``reaggregate_scores()`` utility in ``raki.metrics`` that computes
dataset-level mean scores from per-sample ``SampleResult.scores``,
enabling downstream consumers to re-derive aggregate metrics from saved
JSON reports without re-running the metrics engine.
