"""
access_control.py
-----------------
Uten Assistant — Role-Based Access Control

Defines roles, users, and visibility rules.
Everything access-related lives here so the rest of the codebase
only imports from this one module — a single source of truth.

Visibility model:
    Every chunk stored in pgvector carries a 'visibility' string.
    At retrieval time, the SQL WHERE clause filters by that string.

    "org"              → visible to everyone in the organisation
    "team:{team_id}"   → visible to that team + managers + super_admin
    "private:{user_id}"→ visible only to that user + super_admin

Roles:
    super_admin  can upload (visibility=org), reads everything
    manager      can upload (visibility=team:{id}), reads org + own team + own private
    member       can upload (visibility=private:{id}), reads org + own private
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional



class Role(str, Enum):
    """
    The three roles in Uten.

    Inheriting from str means Role.SUPER_ADMIN == "super_admin" is True —
    you can compare directly with strings from a database or API payload
    without calling .value explicitly.  Very convenient for JWT claims.
    """
    SUPER_ADMIN = "super_admin"
    MANAGER     = "manager"
    MEMBER      = "member"


class Visibility(str, Enum):
    """
    Visibility scope prefixes stored on each chunk.

    We use string prefixes rather than separate columns so the
    WHERE clause stays simple — one column, one filter expression.
    """
    ORG     = "org"           # entire organisation
    TEAM    = "team"          # prefix: "team:{team_id}"
    PRIVATE = "private"       # prefix: "private:{user_id}"



@dataclass
class UtenUser:
    """
    Represents the authenticated user making a request.

    In a real application this would be populated from a JWT token
    or a session object.  You decode the token, extract these fields,
    and pass the UtenUser into every uten.ask() and uten.index_*() call.

    Attributes:
        user_id : unique identifier (UUID or integer from your auth system)
        role    : Role enum value
        team_id : the team this user belongs to (required for managers,
                  optional for members, None for super_admin)
        org_id  : organisation identifier (for multi-tenant deployments)
    """
    user_id: str
    role:    Role
    team_id: Optional[str] = None
    org_id:  str           = "default"

    def is_super_admin(self) -> bool:
        return self.role == Role.SUPER_ADMIN

    def is_manager(self) -> bool:
        return self.role == Role.MANAGER

    def is_member(self) -> bool:
        return self.role == Role.MEMBER

    def __repr__(self) -> str:
        return f"UtenUser(id={self.user_id}, role={self.role.value}, team={self.team_id})"



def make_visibility(user: UtenUser, scope: str = "default") -> str:
    """
    Determine the visibility string to tag a chunk with at index time.

    Args:
        user:  the UtenUser who is uploading the document
        scope: optional override —
               "org"     → force org-wide (only super_admin can do this)
               "team"    → force team scope (manager or super_admin)
               "private" → force private (any role)
               "default" → derive from role automatically

    Returns:
        A visibility string like "org", "team:finance", "private:user_42"

    Raises:
        PermissionError: if the user tries to set a scope above their role.

    Default behaviour by role:
        super_admin → "org"
        manager     → "team:{team_id}"
        member      → "private:{user_id}"
    """
    if scope == "default":
        if user.is_super_admin():
            resolved = Visibility.ORG
        elif user.is_manager():
            resolved = Visibility.TEAM
        else:
            resolved = Visibility.PRIVATE
    else:
        try:
            resolved = Visibility(scope)
        except ValueError:
            raise ValueError(f"Unknown scope '{scope}'. Use: org, team, private")

    if resolved == Visibility.ORG and not user.is_super_admin():
        raise PermissionError(
            f"User {user.user_id} (role={user.role.value}) cannot upload "
            f"with org-wide visibility. Only super_admin can."
        )

    if resolved == Visibility.TEAM and user.is_member():
        raise PermissionError(
            f"User {user.user_id} (role=member) cannot upload with team "
            f"visibility. Use 'private' or ask a manager to upload."
        )

    if resolved == Visibility.ORG:
        return "org"
    elif resolved == Visibility.TEAM:
        if not user.team_id:
            raise ValueError(
                f"User {user.user_id} has no team_id set. "
                f"Assign a team before uploading with team scope."
            )
        return f"team:{user.team_id}"
    else:
        return f"private:{user.user_id}"


def allowed_visibilities(user: UtenUser) -> list[str]:
    """
    Return the list of visibility strings this user is allowed to READ.

    This list is used in the SQL WHERE clause at retrieval time.
    pgvector will only return chunks whose visibility matches one of
    these strings.

    Access matrix:
        super_admin → ["org", "team:*", "private:*"]  (all — wildcard via SQL LIKE)
        manager     → ["org", "team:{their_team}", "private:{their_id}"]
        member      → ["org", "private:{their_id}"]

    Note on super_admin:
        Rather than listing every possible team and user, we handle
        super_admin with a special flag in the SQL query that bypasses
        the visibility filter entirely.  This is cleaner than trying
        to enumerate all possible visibility strings.

    Returns:
        list of exact visibility strings to match, OR an empty list
        which signals "bypass filter" for super_admin.
    """
    if user.is_super_admin():
        return []

    visibilities = ["org", f"private:{user.user_id}"]

    if user.team_id:
        visibilities.append(f"team:{user.team_id}")

    return visibilities



def can_upload(user: UtenUser) -> bool:
    """
    Check whether a user is allowed to upload documents at all.

    All three roles can upload — but what they upload and who sees it
    is controlled by make_visibility().
    """
    return True


def assert_can_upload(user: UtenUser) -> None:
    """Raise PermissionError if user cannot upload. Used as a guard."""
    if not can_upload(user):
        raise PermissionError(
            f"User {user.user_id} (role={user.role.value}) is not permitted to upload."
        )
