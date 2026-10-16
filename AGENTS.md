# Agent Instructions

This project tracks issues in Jira, project AURA at https://aegean-ai.atlassian.net. Use the Atlassian MCP tools.

## Quick Reference

```text
Search issues   mcp__plugin_atlassian_atlassian__searchJiraIssuesUsingJql
View an issue   mcp__plugin_atlassian_atlassian__getJiraIssue
File an issue   mcp__plugin_atlassian_atlassian__createJiraIssue
Comment         mcp__plugin_atlassian_atlassian__addCommentToJiraIssue
Change status   mcp__plugin_atlassian_atlassian__transitionJiraIssue
```

## Session completion

When you finish a chunk of work:

1. File follow-up work in Jira for anything left undone.
2. Run the quality gates if code changed (tests, linters, builds).
3. Update the status of the Jira issues you touched.
4. Leave the work committed or uncommitted according to what the user asked. Do not push on
   your own initiative: the user batches commits deliberately, and several repos protect
   `main` so direct pushes are rejected. Where a repo requires a PR, open one rather than
   pushing to `main`.
5. Hand off: say what is done, what is not, and what the next step is.

## Issue tracking with Jira

Issues live in Jira, project AURA at https://aegean-ai.atlassian.net. Use the Atlassian
MCP tools (`mcp__plugin_atlassian_atlassian__*`) for all issue work.

- Find work with `searchJiraIssuesUsingJql`, read with `getJiraIssue`, file with
  `createJiraIssue`, comment with `addCommentToJiraIssue`, move with `transitionJiraIssue`.
- Before filing, search for an existing issue covering the same work and update that instead.
- When you discover follow-up work mid-task, file it in Jira and link it to the issue you
  are working on rather than leaving a TODO in the code.
- Jira is the only tracker. Do not create markdown TODO lists or a second tracking system.
- There is no local issue database in this repo. beads and the `bd` CLI are retired: never
  run `bd`, never recreate a `.beads/` directory, and ignore any `.beads-archive.zip`.
<!-- END JIRA INTEGRATION -->
