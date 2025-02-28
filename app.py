# app.py - Main Flask application file
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from services.data_service import DataService
from services.sentiment_service import SentimentService
from services.auth_service import AuthService, token_required
from models.database import db, Commodity, SentimentRecord, User
from apscheduler.schedulers.background import BackgroundScheduler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///commodity_sentiment.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key')

# Enable CORS
CORS(app)

# Initialize database
db.init_app(app)

# Initialize services
data_service = DataService()
sentiment_service = SentimentService()
auth_service = AuthService()

# Create database tables
with app.app_context():
    db.create_all()
    # Add default commodities if they don't exist
    default_commodities = ['Oil', 'Gold', 'Silver', 'Copper', 'Natural Gas', 'Wheat', 'Coffee']
    for commodity_name in default_commodities:
        if not Commodity.query.filter_by(name=commodity_name).first():
            commodity = Commodity(name=commodity_name)
            db.session.add(commodity)
    db.session.commit()

# Set up background task to update sentiment
def update_all_sentiments():
    with app.app_context():
        logger.info("Starting scheduled sentiment update")
        commodities = Commodity.query.all()
        for commodity in commodities:
            try:
                for interval in ['hourly', 'daily', 'weekly', 'monthly']:
                    # Get relevant data
                    competition_data = data_service.get_competition_data(commodity.name)
                    price_data = data_service.get_price_data(commodity.name)
                    availability_data = data_service.get_availability_data(commodity.name)
                    demand_data = data_service.get_demand_data(commodity.name)
                    geo_data = data_service.get_geo_data(commodity.name, 'UAE')
                    
                    # Calculate sentiment
                    sentiment = sentiment_service.calculate_sentiment(
                        competition_data, 
                        price_data, 
                        availability_data, 
                        demand_data, 
                        geo_data, 
                        interval
                    )
                    
                    # Save to database
                    new_record = SentimentRecord(
                        commodity_id=commodity.id,
                        interval=interval,
                        sentiment=sentiment,
                        timestamp=datetime.utcnow()
                    )
                    db.session.add(new_record)
                
                db.session.commit()
                logger.info(f"Updated sentiment for {commodity.name}")
            except Exception as e:
                logger.error(f"Error updating sentiment for {commodity.name}: {str(e)}")
                db.session.rollback()
        
        logger.info("Completed scheduled sentiment update")

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_all_sentiments, 'interval', hours=1)

# API routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    return auth_service.register_user(data.get('email'), data.get('password'), data.get('name', ''))

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    return auth_service.login_user(data.get('email'), data.get('password'))

@app.route('/api/commodities', methods=['GET'])
@token_required
def get_commodities():
    commodities = Commodity.query.all()
    return jsonify({
        'commodities': [{'id': c.id, 'name': c.name} for c in commodities]
    })

@app.route('/api/sentiment/<int:commodity_id>', methods=['GET'])
@token_required
def get_sentiment(commodity_id):
    interval = request.args.get('interval', 'daily')
    if interval not in ['hourly', 'daily', 'weekly', 'monthly']:
        return jsonify({'message': 'Invalid interval'}), 400
    
    commodity = Commodity.query.get(commodity_id)
    if not commodity:
        return jsonify({'message': 'Commodity not found'}), 404
    
    # Get the latest sentiment
    sentiment_record = SentimentRecord.query.filter_by(
        commodity_id=commodity_id, 
        interval=interval
    ).order_by(SentimentRecord.timestamp.desc()).first()
    
    if not sentiment_record:
        return jsonify({'message': 'No sentiment data available'}), 404
    
    return jsonify({
        'commodity': commodity.name,
        'interval': interval,
        'sentiment': sentiment_record.sentiment,
        'direction': 'up' if sentiment_record.sentiment > 0 else 'down',
        'timestamp': sentiment_record.timestamp.isoformat()
    })

@app.route('/api/sentiment/history/<int:commodity_id>', methods=['GET'])
@token_required
def get_sentiment_history(commodity_id):
    interval = request.args.get('interval', 'daily')
    days = int(request.args.get('days', 30))
    
    if interval not in ['hourly', 'daily', 'weekly', 'monthly']:
        return jsonify({'message': 'Invalid interval'}), 400
    
    commodity = Commodity.query.get(commodity_id)
    if not commodity:
        return jsonify({'message': 'Commodity not found'}), 404
    
    # Calculate the start date based on days parameter
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get historical sentiment data
    history = SentimentRecord.query.filter(
        SentimentRecord.commodity_id == commodity_id,
        SentimentRecord.interval == interval,
        SentimentRecord.timestamp >= start_date
    ).order_by(SentimentRecord.timestamp.asc()).all()
    
    return jsonify({
        'commodity': commodity.name,
        'interval': interval,
        'history': [
            {
                'sentiment': record.sentiment,
                'direction': 'up' if record.sentiment > 0 else 'down',
                'timestamp': record.timestamp.isoformat()
            } for record in history
        ]
    })

@app.route('/api/dashboard/summary', methods=['GET'])
@token_required
def get_dashboard_summary():
    # Get latest sentiment for all commodities at all intervals
    commodities = Commodity.query.all()
    summary = []
    
    for commodity in commodities:
        commodity_data = {
            'id': commodity.id,
            'name': commodity.name,
            'intervals': {}
        }
        
        for interval in ['hourly', 'daily', 'weekly', 'monthly']:
            sentiment_record = SentimentRecord.query.filter_by(
                commodity_id=commodity.id, 
                interval=interval
            ).order_by(SentimentRecord.timestamp.desc()).first()
            
            if sentiment_record:
                commodity_data['intervals'][interval] = {
                    'sentiment': sentiment_record.sentiment,
                    'direction': 'up' if sentiment_record.sentiment > 0 else 'down',
                    'timestamp': sentiment_record.timestamp.isoformat()
                }
        
        summary.append(commodity_data)
    
    return jsonify({'summary': summary})

# Start the scheduler when the application starts
@app.before_first_request
def start_scheduler():
    scheduler.start()
    logger.info("Started background scheduler")

# Shutdown the scheduler when the application stops
@app.teardown_appcontext
def shutdown_scheduler(exception=None):
    scheduler.shutdown()
    logger.info("Shutdown background scheduler")

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False') == 'True', host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
