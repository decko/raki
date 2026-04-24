package schemas

type PlanOutput struct {
	TicketKey    string         `json:"ticket_key"`
	Approach     string         `json:"approach"`
	Tasks        []PlanTask     `json:"tasks"`
	Verification VerifyStrategy `json:"verification"`
	Deviations   []string       `json:"deviations,omitempty"`
}

type PlanTask struct {
	ID          string   `json:"id"`
	Description string   `json:"description"`
	Files       []string `json:"files"`
	FailingTest string   `json:"failing_test"` // the test to write first
	DoneWhen    string   `json:"done_when"`
	DependsOn   []string `json:"depends_on,omitempty"`
}

type VerifyStrategy struct {
	Commands    []string `json:"commands"`
	ManualSteps []string `json:"manual_steps,omitempty"`
}
