package schemas

type TriageOutput struct {
	TicketKey    string   `json:"ticket_key"`
	CodeArea     string   `json:"code_area"`
	Files        []string `json:"files"`
	Complexity   string   `json:"complexity"` // small, medium, large
	Approach     string   `json:"approach"`
	Risks        []string `json:"risks"`
	Specialists  []string `json:"specialists"` // python, security, rag, doc
	Automatable  bool     `json:"automatable"`
	BlockReason  string   `json:"block_reason,omitempty"`
}
