# models/database.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def __init__(self, email, password, name=''):
        self.email = email
        self.password = password
        self.name = name
        self.public_id = str(uuid.uuid4())

class Commodity(db.Model):
    __tablename__ = 'commodities'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    sentiment_records = db.relationship('SentimentRecord', backref='commodity', lazy=True)

class SentimentRecord(db.Model):
    __tablename__ = 'sentiment_records'
    
    id = db.Column(db.Integer, primary_key=True)
    commodity_id = db.Column(db.Integer, db.ForeignKey('commodities.id'), nullable=False)
    interval = db.Column(db.String(20), nullable=False)  # hourly, daily, weekly, monthly
    sentiment = db.Column(db.Float, nullable=False)  # -1.0 to 1.0
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Indices
    __table_args__ = (
        db.Index('idx_commodity_interval_timestamp', 'commodity_id', 'interval', 'timestamp'),
    )
