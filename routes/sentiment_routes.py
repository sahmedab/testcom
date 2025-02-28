# routes/sentiment_routes.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from database import get_db
from services.sentiment_service import get_sentiment_service, SentimentAnalysisService
from models.commodity_model import Commodity
from schemas.sentiment_schema import (
    SentimentResponse, 
    SentimentHistoryResponse,
    SentimentAccuracyResponse
)

router = APIRouter(
    prefix="/api/sentiment",
    tags=["sentiment"]
)

@router.get("/prediction/{commodity_id}", response_model=SentimentResponse)
async def get_commodity_sentiment(
    commodity_id: int,
    target_country: str = "UAE",
    db: Session = Depends(get_db),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
):
    """
    Generate sentiment predictions for a specific commodity across different time intervals.
    
    - **commodity_id**: ID of the commodity to analyze
    - **target_country**: Target country for analysis (default: UAE)
    """
    try:
        # Check if commodity exists
        commodity = db.query(Commodity).filter(Commodity.id == commodity_id).first()
        if not commodity:
            raise HTTPException(status_code=404, detail=f"Commodity with ID {commodity_id} not found")
            
        # Generate sentiment prediction
        sentiment = await sentiment_service.generate_sentiment(commodity_id, target_country)
        
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{commodity_id}", response_model=List[SentimentHistoryResponse])
async def get_sentiment_history(
    commodity_id: int,
    timeframe: str = Query(..., description="Analysis timeframe: hourly, daily, weekly, or monthly"),
    days: int = Query(30, description="Number of days of history to retrieve"),
    db: Session = Depends(get_db),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
):
    """
    Get historical sentiment predictions for a commodity.
    
    - **commodity_id**: ID of the commodity
    - **timeframe**: Timeframe for analysis (hourly, daily, weekly, monthly)
    - **days**: Number of days of history to retrieve (default: 30)
    """
    try:
        # Validate timeframe
        valid_timeframes = ["hourly", "daily", "weekly", "monthly"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )
            
        # Check if commodity exists
        commodity = db.query(Commodity).filter(Commodity.id == commodity_id).first()
        if not commodity:
            raise HTTPException(status_code=404, detail=f"Commodity with ID {commodity_id} not found")
            
        # Get history
        history = await sentiment_service.get_sentiment_history(commodity_id, timeframe, days)
        
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/accuracy/{commodity_id}", response_model=SentimentAccuracyResponse)
async def get_sentiment_accuracy(
    commodity_id: int,
    timeframe: str = Query(..., description="Analysis timeframe: hourly, daily, weekly, or monthly"),
    days: int = Query(30, description="Number of days of history to analyze"),
    db: Session = Depends(get_db),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
):
    """
    Calculate accuracy of past sentiment predictions.
    
    - **commodity_id**: ID of the commodity
    - **timeframe**: Timeframe for analysis (hourly, daily, weekly, monthly)
    - **days**: Number of days of history to analyze (default: 30)
    """
    try:
        # Validate timeframe
        valid_timeframes = ["hourly", "daily", "weekly", "monthly"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )
            
        # Check if commodity exists
        commodity = db.query(Commodity).filter(Commodity.id == commodity_id).first()
        if not commodity:
            raise HTTPException(status_code=404, detail=f"Commodity with ID {commodity_id} not found")
            
        # Get accuracy
        accuracy = await sentiment_service.get_sentiment_accuracy(commodity_id, timeframe, days)
        
        return accuracy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch", response_model=List[SentimentResponse])
async def get_batch_sentiment(
    commodity_ids: str = Query(..., description="Comma-separated list of commodity IDs"),
    target_country: str = "UAE",
    db: Session = Depends(get_db),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
):
    """
    Generate sentiment predictions for multiple commodities.
    
    - **commodity_ids**: Comma-separated list of commodity IDs to analyze
    - **target_country**: Target country for analysis (default: UAE)
    """
    try:
        # Parse commodity IDs
        id_list = [int(id_str) for id_str in commodity_ids.split(",") if id_str.strip()]
        
        if not id_list:
            raise HTTPException(status_code=400, detail="No valid commodity IDs provided")
            
        # Generate predictions for each commodity
        results = []
        for commodity_id in id_list:
            try:
                # Check if commodity exists
                commodity = db.query(Commodity).filter(Commodity.id == commodity_id).first()
                if not commodity:
                    results.append({
                        "commodity_id": commodity_id,
                        "error": f"Commodity with ID {commodity_id} not found"
                    })
                    continue
                    
                # Generate sentiment
                sentiment = await sentiment_service.generate_sentiment(commodity_id, target_country)
                results.append(sentiment)
                
            except Exception as e:
                results.append({
                    "commodity_id": commodity_id,
                    "error": str(e)
                })
                
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
