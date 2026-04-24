package schemas

type ImplementOutput struct {
	TicketKey    string         `json:"ticket_key"`
	Branch       string         `json:"branch"`
	Commits      []CommitRecord `json:"commits"`
	FilesChanged []FileChange   `json:"files_changed"`
	TaskResults  []TaskResult   `json:"task_results"`
	TestsPassed  bool           `json:"tests_passed"`
	TestOutput   string         `json:"test_output,omitempty"`
	Deviations   []string       `json:"deviations,omitempty"`
}

type CommitRecord struct {
	Hash    string `json:"hash"`
	Message string `json:"message"`
	TaskID  string `json:"task_id"`
}

type FileChange struct {
	Path   string `json:"path"`
	Action string `json:"action"` // created, modified, deleted
}

type TaskResult struct {
	TaskID string `json:"task_id"`
	Status string `json:"status"` // completed, failed, skipped
	Reason string `json:"reason,omitempty"`
}
