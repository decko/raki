package schemas

type VerifyOutput struct {
	TicketKey       string            `json:"ticket_key"`
	Verdict         string            `json:"verdict"` // PASS, FAIL
	CriteriaResults []CriterionResult `json:"criteria_results"`
	CommandResults  []CommandResult   `json:"command_results"`
	CodeIssues      []CodeIssue       `json:"code_issues,omitempty"`
	RakiChecks      RakiChecks        `json:"raki_checks"`
	FixesRequired   []string          `json:"fixes_required,omitempty"`
}

type CriterionResult struct {
	Criterion string `json:"criterion"`
	Passed    bool   `json:"passed"`
	Evidence  string `json:"evidence"`
}

type CommandResult struct {
	Command  string `json:"command"`
	ExitCode int    `json:"exit_code"`
	Output   string `json:"output"`
	Passed   bool   `json:"passed"`
}

type CodeIssue struct {
	File     string `json:"file"`
	Line     int    `json:"line,omitempty"`
	Severity string `json:"severity"` // critical, major, minor
	Issue    string `json:"issue"`
}

// RakiChecks are project-specific verification checks.
type RakiChecks struct {
	MetricRegistered  bool `json:"metric_registered"`   // in ALL_OPERATIONAL + METRIC_METADATA + OPERATIONAL_METRICS
	NASemantics       bool `json:"na_semantics"`         // zero-denominator returns None
	RedactionApplied  bool `json:"redaction_applied"`    // redact_sensitive() on new content paths
	PathSafety        bool `json:"path_safety"`          // traversal guard on new path handling
	VersionConsistent bool `json:"version_consistent"`   // pyproject.toml matches __init__.py
}
