"""Safe, privilege-aware RAG demo.

The pipeline, in order:

    user query
      -> input guardrails      (injection / size checks)      guardrails.py
      -> ACL pre-filter        (what may this user see?)      policy.py
      -> retrieval             (only over permitted chunks)   retriever.py
      -> context sanitising    (docs are data, not orders)    guardrails.py
      -> LLM                   (stub by default)              llm.py
      -> output guardrails     (citations, PII, canary)       guardrails.py
      -> audit log             (who saw what, and why)        audit.py

The one rule everything else hangs off: the model must never be shown a
chunk the user is not allowed to read. Filtering after generation is too
late -- the model has already seen it and can leak it.
"""
