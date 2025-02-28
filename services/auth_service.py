# services/auth_service.py
from flask import jsonify, request
from functools import wraps
import jwt
import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import os
from models.database import db, User

class AuthService:
    def register_user(self, email, password, name=''):
        # Check if user already exists
        user = User.query.filter_by(email=email).first()
        if user:
            return jsonify({'message': 'User already exists'}), 409
        
        # Create new user
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(email=email, password=hashed_password, name=name)
        
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({'message': 'User registered successfully'}), 201
    
    def login_user(self, email, password):
        user = User.query.filter_by(email=email).first()
        
        if not user:
            return jsonify({'message': 'Invalid credentials'}), 401
        
        if check_password_hash(user.password, password):
            # Generate JWT token
            token = jwt.encode({
                'public_id': user.public_id,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
            }, os.environ.get('SECRET_KEY', 'dev_secret_key'))
            
            # Update last login timestamp
            user.last_login = datetime.datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                'token': token,
                'user': {
                    'id': user.public_id,
                    'name': user.name,
                    'email': user.email
                }
            })
        
        return jsonify({'message': 'Invalid credentials'}), 401

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, os.environ.get('SECRET_KEY', 'dev_secret_key'), algorithms=["HS256"])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(*args, **kwargs)
    
    return decorated
