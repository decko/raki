Add ``AlcovePipelineAdapter`` for Alcove pipeline-run exports.

Detects and loads directories produced by an Alcove pipeline export
(``run.json`` + ``steps/`` sub-directory).  Each pipeline step becomes a
:class:`PhaseResult`; review findings are parsed from the semicolon-delimited
``outputs.issues`` field; rework cycles are counted from corrective step
activation; cost is aggregated across all step transcripts.

Registered before ``AlcoveAdapter`` in the default registry so that pipeline
export directories are correctly identified before the single-file adapter is
tried.
