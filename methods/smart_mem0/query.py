"""Compatibility facade for the active answer lifecycle and rendering."""

from .query_answer_runtime import QueryAnswerRuntimeMixin, QUERY_TOKEN_STAGES
from .query_rendering import QueryRenderingMixin


class QueryMixin(QueryRenderingMixin, QueryAnswerRuntimeMixin):
    pass
