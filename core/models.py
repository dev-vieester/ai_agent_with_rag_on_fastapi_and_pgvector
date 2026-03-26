from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


def new_uuid() -> str:
    return str(uuid.uuid4())


class Organisation(Base):
    __tablename__ = "organisations"

    id = Column(String(36), primary_key=True, default=new_uuid)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    teams = relationship("Team", back_populates="organisation", cascade="all, delete-orphan")
    users = relationship("User", back_populates="organisation", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="organisation", cascade="all, delete-orphan")


class Team(Base):
    __tablename__ = "teams"

    id = Column(String(36), primary_key=True, default=new_uuid)
    name = Column(String(200), nullable=False)
    org_id = Column(String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("org_id", "name", name="uq_team_org_name"),
    )

    organisation = relationship("Organisation", back_populates="teams")
    members = relationship("User", back_populates="team")


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=new_uuid)
    email = Column(String(300), nullable=False, unique=True, index=True)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(50), nullable=False, default="member")
    team_id = Column(String(36), ForeignKey("teams.id", ondelete="SET NULL"), nullable=True, index=True)
    org_id = Column(String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    account_status = Column(String(20), default="pending", nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

    organisation = relationship("Organisation", back_populates="users")
    team = relationship("Team", back_populates="members")
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=new_uuid)
    filename = Column(String(500), nullable=False, index=True)
    file_type = Column(String(50), nullable=False)
    visibility = Column(String(100), nullable=False, default="org", index=True)
    owner_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    org_id = Column(String(36), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    owner = relationship("User", back_populates="documents")
    organisation = relationship("Organisation", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    document = relationship("Document", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    embedding = Column(Vector(384), nullable=False)

    chunk = relationship("Chunk", back_populates="embedding")
