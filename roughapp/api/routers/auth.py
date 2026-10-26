from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from roughapp.auth import (
    hash_password, verify_password, create_token, get_current_user,
    get_guest_session_optional, adopt_guest_session, clear_guest_cookie,
)
from roughapp.db.engine import get_db
from roughapp.db.models import GuestSession, User

router = APIRouter(prefix="/auth")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(
    req: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db),
    guest_session: GuestSession | None = Depends(get_guest_session_optional),
):
    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")
    if db.query(User).filter_by(email=req.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=req.email, password_hash=hash_password(req.password))
    db.add(user)
    db.flush()
    if guest_session is not None:
        adopt_guest_session(db, guest_session, user)
        clear_guest_cookie(response)
    db.commit()
    db.refresh(user)
    return TokenResponse(access_token=create_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(
    req: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
    guest_session: GuestSession | None = Depends(get_guest_session_optional),
):
    user = db.query(User).filter_by(email=req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if guest_session is not None:
        adopt_guest_session(db, guest_session, user)
        clear_guest_cookie(response)
        db.commit()
    return TokenResponse(access_token=create_token(user.id))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email)
