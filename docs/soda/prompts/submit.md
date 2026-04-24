You are submitting a verified and reviewed implementation as a pull request for the RAKI project.

## Ticket

Key: {{.Ticket.Key}}
Summary: {{.Ticket.Summary}}

## Implementation Report
{{.Artifacts.Implement}}

## Verification Report
{{.Artifacts.Verify}}

## Review Report
{{.Artifacts.Review}}

## Working Directory

Worktree: {{.WorktreePath}}
Branch: {{.Branch}}

## Your Task

1. **Push the branch** to origin.
2. **Create the PR** with `gh pr create`:
   - Title: `<type>(<scope>): <subject>` (under 70 chars)
   - Body includes: summary, acceptance criteria checklist, review results per specialist
   - Add label `ai-assisted`
   - Reference the ticket: `Refs #<ticket-number>`
   - Do NOT use `Closes` — issues are closed manually after human review
   - Include trailers:
     ```
     Assisted-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
     Assigned-by: decko
     ```

3. **Create a towncrier changelog fragment** before pushing:
   - Determine the fragment type: `feature` (new capability), `fix` (bug fix), `breaking` (breaking change), `doc` (documentation), `misc` (other)
   - Create the file `changes/<ticket-number>.<type>` with a one-line description of the user-facing change
   - Example: `echo "Add --gate and --require-metric flags to report subcommand." > changes/139.fix`
   - Stage and commit the fragment: `git add changes/ && git commit -m "chore: add towncrier fragment for #<ticket-number>"`

4. **Do NOT merge** — the PR is for human review. Report the PR URL.
