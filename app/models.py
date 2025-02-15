from sqlalchemy import Column, Integer, DateTime, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Person(Base):
    __tablename__ = 'persons'
    
    person_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Detection(Base):
    __tablename__ = 'detections'
    
    detection_id = Column(Integer, primary_key=True)
    person_id = Column(UUID(as_uuid=True), ForeignKey('persons.person_id'))
    camera_id = Column(String(50))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    embedding_id = Column(String(100))  # Reference to Milvus ID