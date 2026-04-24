package schemas

// ReviewOutput is the structured output for the multi-specialist review phase.
type ReviewOutput struct {
	TicketKey      string             `json:"ticket_key"`
	Verdict        string             `json:"verdict"` // approve, rework
	Perspectives   []PerspectiveResult `json:"perspectives"`
	RoutingReason  string             `json:"routing_reason,omitempty"` // why rework was triggered
}

// PerspectiveResult is the review from one specialist perspective.
type PerspectiveResult struct {
	Name     string          `json:"name"` // python, security, rag, doc
	Verdict  string          `json:"verdict"` // clean, needs_fixes
	Findings []ReviewFinding `json:"findings"`
}

// ReviewFinding is a single issue found during specialist review.
type ReviewFinding struct {
	Severity   string `json:"severity"` // CRITICAL, IMPORTANT, MINOR
	File       string `json:"file"`
	Line       int    `json:"line,omitempty"`
	Issue      string `json:"issue"`
	Suggestion string `json:"suggestion,omitempty"`
}
