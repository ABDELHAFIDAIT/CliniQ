from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional


class UserBase(BaseModel) :
    email : EmailStr
    username : str
    role : Optional[str] = "medecin"


class UserCreate(UserBase) :
    password : str
    

class UserUpdate(BaseModel) :
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    

class UserOut(UserBase) :
    id : int
    
    model_config = ConfigDict(from_attributes=True)