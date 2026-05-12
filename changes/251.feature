Add `triage_calibration` metric — measures how well the agent's triage complexity prediction
(`small` / `medium` / `large`) matches the actual session cost. Calibration thresholds:
small ≤ $8.00, medium ≤ $16.00, large: any cost. Returns N/A when no sessions have both a
triage complexity estimate and a cost.
