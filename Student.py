# -*- coding: utf-8 -*-
"""
Created on April 27 2024
@authors: E. ROJAS - K. QUISPE 
"""
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    gender: int
    age: int
    birthDistrict: int
    currentDistrict: int
    civilStatus: int
    typeInstitution: int
    provenanceLevel: int
    disability: int
    enrolledCycle: int
    repeatedCourses: int
    languageLevel: int
    computingLevel: int
    isForeign: int