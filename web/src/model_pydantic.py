from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    username: str
    password: str

class PatientCreate(BaseModel):
    name: str
    age: int
    sex: str
    glucose: float
    bmi: float
    bloodpressure: float
    pedigree: float