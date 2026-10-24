"""add semantic layer

Revision ID: 669e5f3dc4d3
Revises: 1bc65786c974
Create Date: 2026-10-24 20:24:52.783224

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "669e5f3dc4d3"
down_revision: str | Sequence[str] | None = "1bc65786c974"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Article profile from scripts/enrich.py; empty until enrichment runs.
    op.add_column("documents", sa.Column("topic", sa.Text(), nullable=True))
    op.add_column("documents", sa.Column("summary", sa.Text(), nullable=True))

    op.create_table(
        "entities",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),  # person | place | organization | event
        sa.Column(
            "aliases",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.UniqueConstraint("name", "type", name="uq_entities_name_type"),
    )

    op.create_table(
        "chunk_entities",
        sa.Column(
            "chunk_id",
            sa.BigInteger(),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "entity_id",
            sa.BigInteger(),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    op.create_index("ix_chunk_entities_entity", "chunk_entities", ["entity_id"])


def downgrade() -> None:
    op.drop_index("ix_chunk_entities_entity", table_name="chunk_entities")
    op.drop_table("chunk_entities")
    op.drop_table("entities")
    op.drop_column("documents", "summary")
    op.drop_column("documents", "topic")
