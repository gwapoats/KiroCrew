"""LOCAL PATCHES (not upstream): owner-configured extensions kept out of upstream files.

Every module here is read from the top-level ``local_ext`` section of
``~/.kiro/crew/config.json``. With that section absent, nothing here changes any
behaviour. Upstream files reach these modules only through short, marked call sites
(``LOCAL PATCH (local_ext...)``) so the patch series rebases with small conflicts.
"""
