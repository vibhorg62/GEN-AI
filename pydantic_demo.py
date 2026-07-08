from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Vasu' # default value
    age: Optional[int] = None # optional field
    email:EmailStr
    cgpa:float = Field(gt=0,lt=10,default=5,description="Cumulative Grade Point Average") # gt means greater than and lt means less than
    
new_student = {'name':'Vibhor','age':21,'email':'vasu@gmail.com','cgpa':8.58} # yahan denge toh yeh chlega, agar nahi denge toh default value 'Vasu' hi print hoga , gmail validation bhi ho rahi

student = Student(**new_student)

student_dict = dict(student) # converting pydantic model to dictionary
print(student_dict['age']) # printing the dictionary

student_json=student.model_dump_json() # converting pydantic model to json
print(student_json) # printing the json