"""Create companies table

Revision ID: xxxx_create_companies
Revises: yyyy_previous_migration
Create Date: 2025-05-13
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime


def upgrade():
    # companies テーブルの作成
    op.create_table(
        'companies',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('industry', sa.String(), nullable=True),
        sa.Column('founded_date', sa.DateTime(), nullable=True),
        sa.Column('employee_count', sa.Integer(), nullable=True),
        sa.Column('location', sa.String(), nullable=True),
        sa.Column('website', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_companies_name'), 'companies', ['name'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_companies_name'), table_name='companies')
    op.drop_table('companies')