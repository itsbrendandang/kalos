"""Ingestion-driven closed loop: a push DataFeed + ProposalSink, anonymized on read."""
from .feed import anonymize_frame, DataFeed, ProposalSink, FileDataFeed, FileProposalSink
from .runner import Problem, propose_next_batch, IngestionLoop

__all__ = [
    "anonymize_frame", "DataFeed", "ProposalSink", "FileDataFeed", "FileProposalSink",
    "Problem", "propose_next_batch", "IngestionLoop",
]
