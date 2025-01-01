from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Table
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class UserRole(str, Enum):
    """ユーザーロール"""
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class GroupMember(Base):
    """グループメンバーモデル"""
    __tablename__ = 'group_members'

    group_id = Column(String, ForeignKey('groups.id'), primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), primary_key=True)
    role = Column(String, nullable=False)  # admin, member, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    group = relationship("Group", back_populates="group_members")
    user = relationship("User", back_populates="group_memberships")

group_tags = Table(
    'group_tags',
    Base.metadata,
    Column('group_id', String, ForeignKey('groups.id')),
    Column('tag_id', String, ForeignKey('tags.id')),
)

class User(Base):
    """ユーザーモデル"""
    __tablename__ = 'users'

    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # admin, user, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    owned_companies = relationship("Company", back_populates="owner")
    owned_groups = relationship("Group", back_populates="owner")
    group_memberships = relationship("GroupMember", back_populates="user")

class Company(Base):
    """会社モデル"""
    __tablename__ = 'companies'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    industry = Column(String)
    owner_id = Column(String, ForeignKey('users.id'), nullable=False)
    founded_date = Column(DateTime)
    employee_count = Column(Integer)
    website = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    owner = relationship("User", back_populates="owned_companies")
    statuses = relationship("Status", back_populates="company", cascade="all, delete-orphan")
    stages = relationship("Stage", back_populates="company", cascade="all, delete-orphan")

class Status(Base):
    """会社のステータスモデル"""
    __tablename__ = 'statuses'

    id = Column(String, primary_key=True)
    company_id = Column(String, ForeignKey('companies.id'), nullable=False)
    type = Column(String, nullable=False)  # ACTIVE, INACTIVE, etc.
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # リレーションシップ
    company = relationship("Company", back_populates="statuses")

class Stage(Base):
    """会社のステージモデル"""
    __tablename__ = 'stages'

    id = Column(String, primary_key=True)
    company_id = Column(String, ForeignKey('companies.id'), nullable=False)
    type = Column(String, nullable=False)  # SEED, SERIES_A, etc.
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # リレーションシップ
    company = relationship("Company", back_populates="stages")

class Group(Base):
    """グループモデル"""
    __tablename__ = 'groups'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    owner_id = Column(String, ForeignKey('users.id'), nullable=False)
    is_private = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    owner = relationship("User", back_populates="owned_groups")
    group_members = relationship("GroupMember", back_populates="group")
    tags = relationship(
        "Tag",
        secondary=group_tags,
        back_populates="groups"
    )

class Tag(Base):
    """タグモデル"""
    __tablename__ = 'tags'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # リレーションシップ
    groups = relationship(
        "Group",
        secondary=group_tags,
        back_populates="tags"
    )

class ReportTemplate(Base):
    """レポートテンプレートモデル"""
    __tablename__ = 'report_templates'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    format = Column(String, nullable=False)  # pdf, docx, etc.
    template_content = Column(String, nullable=False)
    variables = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    reports = relationship("Report", back_populates="template")

class Report(Base):
    """レポートモデル"""
    __tablename__ = 'reports'

    id = Column(String, primary_key=True)
    template_id = Column(String, ForeignKey('report_templates.id'), nullable=False)
    content = Column(String, nullable=False)
    format = Column(String, nullable=False)  # pdf, docx, etc.
    report_metadata = Column(JSON)  # metadata を report_metadata に変更
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # リレーションシップ
    template = relationship("ReportTemplate", back_populates="reports")