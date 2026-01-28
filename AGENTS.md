# Agent Rules

- Never implement code directly; only suggest inline changes.
- Work one function at a time; dont do too much at once.
- Always explain what you are doing.
- Read and understand the existing code before advising.
- Use `rg` for searches when needed.
- Models must be defined in `pufferlib/models.py`.
- Keep edits ASCII unless the file already uses Unicode.
- Dont use destructive git commands or revert unrelated changes.
- Use apply_patch only for single-file edits; otherwise suggest inline changes.
- When asked for a plan, make it multi-step and update it as we go.
- If you notice unexpected changes, stop and ask how to proceed.

