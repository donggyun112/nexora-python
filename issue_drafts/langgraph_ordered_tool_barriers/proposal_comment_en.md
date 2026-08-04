I have a related but slightly different use case. I do not need every tool call to run sequentially. For a batch like `read_a, read_b, write_a, write_b, read_c`, I would want `(read_a || read_b) -> write_a -> write_b -> read_c`: consecutive read-only calls can overlap, while mutating or exclusive calls act as ordered barriers and cannot be overtaken.

Today `ToolNode` starts every call in the batch together. `parallel_tool_calls=False` adds model turns, while `max_concurrency=1` also serializes the safe reads, so neither expresses this policy.

Would an opt-in per-tool `shared`/`exclusive` execution mode, or a classifier callback, be in scope for `ToolNode`? The default could remain unchanged. This request is about ordering external side effects; propagating `Command` state between calls can remain a separate concern.
