import logging
import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, EmailStr
from backend.utils import config
from backend.utils.database import get_collection
from backend.utils.mongo import serialize_document, make_response

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger("AuthAPI")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

# Pydantic schemas
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ForgotPassword(BaseModel):
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

# Database helper functions
def get_user(email: str):
    users_col = get_collection("users")
    doc = users_col.find_one({"email": email})
    return serialize_document(doc)

def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        pwd_bytes = plain_password.encode('utf-8')[:72]
        hash_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = get_user(email)
    if user is None:
        raise credentials_exception
    return user

# Routes
@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    try:
        existing_user = get_user(user_data.email)
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
            
        hashed_pwd = hash_password(user_data.password)
        users_col = get_collection("users")
        
        new_user = {
            "username": user_data.username,
            "email": user_data.email,
            "hashed_password": hashed_pwd,
            "created_at": datetime.utcnow().isoformat()
        }
        users_col.insert_one(serialize_document(new_user))
        
        access_token = create_access_token(
            data={"sub": user_data.email},
            expires_delta=timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        res_data = {"access_token": access_token, "token_type": "bearer", "username": user_data.username}
        return make_response(data=res_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error during registration for {user_data.email}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Registration failed: {str(e)}")


@router.post("/login", response_model=dict)
async def login(user_data: UserLogin):
    try:
        user = get_user(user_data.email)
        if not user or not verify_password(user_data.password, user["hashed_password"]):
            raise HTTPException(status_code=400, detail="Incorrect email or password")
            
        access_token = create_access_token(
            data={"sub": user["email"]},
            expires_delta=timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        res_data = {"access_token": access_token, "token_type": "bearer", "username": user["username"]}
        return make_response(data=res_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forgot-password", response_model=dict)
async def forgot_password(data: ForgotPassword):
    try:
        user = get_user(data.email)
        if not user:
            raise HTTPException(status_code=404, detail="Email not found")
        
        res_data = {"message": f"Password reset instructions sent to {data.email}"}
        return make_response(data=res_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# Special OAuth2 endpoint for Swagger UI testing
@router.post("/token", response_model=dict)
async def login_for_swagger(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = get_user(form_data.username)  # Form data uses 'username' field for email or username
        if not user:
            # Try finding by username instead of email
            users_col = get_collection("users")
            user = users_col.find_one({"username": form_data.username})
            user = serialize_document(user)
            
        if not user or not verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username/email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        access_token = create_access_token(
            data={"sub": user["email"]},
            expires_delta=timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        res_data = {"access_token": access_token, "token_type": "bearer", "username": user["username"]}
        return make_response(data=res_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/me", response_model=dict)
async def get_me(current_user: dict = Depends(get_current_user)):
    try:
        user_copy = current_user.copy()
        if "hashed_password" in user_copy:
            del user_copy["hashed_password"]
        return make_response(data=user_copy)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

