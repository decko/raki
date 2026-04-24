package schemas

type SubmitOutput struct {
	TicketKey string `json:"ticket_key"`
	PRURL     string `json:"pr_url"`
	PRNumber  int    `json:"pr_number"`
	Title     string `json:"title"`
	Branch    string `json:"branch"`
	Target    string `json:"target"`
}
