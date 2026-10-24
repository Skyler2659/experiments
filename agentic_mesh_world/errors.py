class Real2CodeError(Exception):
    """An input, evidence, or world-contract error."""


class VersionConflict(Real2CodeError):
    """A write was based on an obsolete world version."""


class UnsupportedQuery(Real2CodeError):
    """The active backend cannot answer this query."""


class StopGateClosed(Real2CodeError):
    """A confirmed simulation step-boundary stop rejected a later command."""


class PlanningBudgetExceeded(Real2CodeError):
    """A registered planning query would exceed, or has exceeded, its budget."""
