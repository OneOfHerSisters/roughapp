from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import secrets
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from roughapp.config import get_settings
from roughapp.db.engine import get_db
from roughapp.db.guest_sessions import cleanup_expired_guest_sessions
from roughapp.db.models import Dataset, Experiment, GuestSession, User, ensure_utc, utcnow

settings = get_settings()

SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = settings.access_token_expire_days
GUEST_SESSION_EXPIRE_HOURS = settings.guest_session_expire_hours
GUEST_COOKIE_NAME = "roughapp_guest"

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: int) -> str:
    expire = utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": str(user_id), "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> Optional[int]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None


def _hash_guest_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _guest_expiry() -> datetime:
    return utcnow() + timedelta(hours=GUEST_SESSION_EXPIRE_HOURS)


def _set_guest_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=GUEST_COOKIE_NAME,
        value=token,
        max_age=GUEST_SESSION_EXPIRE_HOURS * 3600,
        httponly=True,
        samesite="lax",
    )


def clear_guest_cookie(response: Response) -> None:
    response.delete_cookie(GUEST_COOKIE_NAME)


@dataclass
class RequestOwner:
    # HTTP-side identity (knows about JWT/cookies); mapped to the
    # transport-agnostic business ActorContext in api/dependencies.py.
    user: Optional[User] = None
    guest_session: Optional[GuestSession] = None

    @property
    def is_guest(self) -> bool:
        return self.user is None and self.guest_session is not None

    @property
    def has_owner(self) -> bool:
        return self.user is not None or self.guest_session is not None


def create_guest_session(db: Session, response: Response) -> GuestSession:
    token = secrets.token_urlsafe(32)
    session = GuestSession(token_hash=_hash_guest_token(token), expires_at=_guest_expiry())
    db.add(session)
    db.flush()
    _set_guest_cookie(response, token)
    return session


def get_guest_session_optional(
    guest_token: Optional[str] = Cookie(default=None, alias=GUEST_COOKIE_NAME),
    db: Session = Depends(get_db),
) -> Optional[GuestSession]:
    if not guest_token:
        return None
    session = db.query(GuestSession).filter_by(token_hash=_hash_guest_token(guest_token)).first()
    if not session:
        return None
    if ensure_utc(session.expires_at) <= utcnow():
        cleanup_expired_guest_sessions(db)
        return None
    return session


def ensure_owner(owner: RequestOwner, db: Session, response: Response) -> RequestOwner:
    if owner.has_owner:
        if owner.guest_session is not None:
            owner.guest_session.expires_at = _guest_expiry()
        return owner
    guest_session = create_guest_session(db, response)
    return RequestOwner(guest_session=guest_session)


def adopt_guest_session(db: Session, guest_session: GuestSession, user: User) -> None:
    db.query(Dataset).filter(Dataset.guest_session_id == guest_session.id).update(
        {Dataset.user_id: user.id, Dataset.guest_session_id: None},
        synchronize_session=False,
    )
    db.query(Experiment).filter(Experiment.guest_session_id == guest_session.id).update(
        {Experiment.user_id: user.id, Experiment.guest_session_id: None},
        synchronize_session=False,
    )
    db.delete(guest_session)


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if not credentials:
        return None
    user_id = _decode_token(credentials.credentials)
    if not user_id:
        return None
    return db.get(User, user_id)


def get_request_owner_optional(
    user: Optional[User] = Depends(get_current_user_optional),
    guest_session: Optional[GuestSession] = Depends(get_guest_session_optional),
) -> RequestOwner:
    if user is not None:
        return RequestOwner(user=user)
    return RequestOwner(guest_session=guest_session)


def get_current_user(
    user: Optional[User] = Depends(get_current_user_optional),
) -> User:
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user
