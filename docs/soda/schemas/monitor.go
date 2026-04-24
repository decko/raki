package schemas

type MonitorOutput struct {
	TicketKey       string          `json:"ticket_key"`
	PRURL           string          `json:"pr_url"`
	CommentsHandled []CommentAction `json:"comments_handled"`
	FilesChanged    []FileChange    `json:"files_changed,omitempty"`
	Commits         []CommitRecord  `json:"commits,omitempty"`
	TestsPassed     bool            `json:"tests_passed"`
}

type CommentAction struct {
	CommentID string `json:"comment_id"`
	Author    string `json:"author"`
	Content   string `json:"content"`
	Action    string `json:"action"` // fixed, explained, deferred
	Response  string `json:"response"`
}
