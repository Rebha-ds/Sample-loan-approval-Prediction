from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from app.config.database import get_db
from app.features.authentication.components.auth_service import (
    register_user,
    login_user
)
from app.features.authentication.models.user_model import User

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


class RegisterSchema(BaseModel):
    full_name: str
    email: EmailStr
    password: str


class LoginSchema(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
def register(
    payload: RegisterSchema,
    db: Session = Depends(get_db)
):
    return register_user(payload, db, User)


@router.post("/login")
def login(
    payload: LoginSchema,
    db: Session = Depends(get_db)
):
    return login_user(
        payload.email,
        payload.password,
        db,
        User
    )