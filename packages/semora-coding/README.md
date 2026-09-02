# semora-coding

The coding-agent layer over the `semora` core: built-in tools (`read`, `write`, `edit`, `grep`,
`glob`, `Bash`, `web_fetch`), system-prompt assembly, plan mode, goals, the skill catalog and
deferred tool search.

```
uv add "semora[coding]"
```

It is a reference assembly. The core promises that a tool's effect happens once and that every
decision about it has a seam; this package is one answer to what those tools and seams say to a
model. Replace any of it — the core does not know it is here.
