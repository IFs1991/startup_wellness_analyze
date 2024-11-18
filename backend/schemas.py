# -*- coding: utf-8 -*-
"""
Pydantic スキーマ

API リクエストやレスポンスのデータ構造を定義する Pydantic モデルを定義します。

"""

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    hashed_password: str

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str