# models.py
from sqlalchemy import Column, Integer, DateTime, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TrackingRecord(Base):
    __tablename__ = 'tracking_records'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    camera_id = Column(String)
    embedding_id = Column(String)
