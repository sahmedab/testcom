# services/sentiment_service.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import Dict, List, Optional, Tuple, Union
import json
from sqlalchemy.orm import Session

from models.sentiment_model import SentimentPrediction
from models.commodity_model import Commodity
from data_service import data_service
from config import settings
from utils.geography import calculate_shipping_impact

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysisService:
    """Service for analyzing data and generating sentiment predictions for commodity prices."""
    
    def __init__(self, db_session: Session):
        """Initialize the sentiment analysis service with required dependencies."""
        self.db_session = db_session
        self.data_service = data_service.DataService(db_session)
        
    async def generate_sentiment(self, commodity_id: int, target_country: str = "UAE") -> Dict:
        """
        Generate sentiment predictions for a specific commodity across different time intervals.
        
        Args:
            commodity_id: The ID of the commodity to analyze
            target_country: The target country for analysis (default: UAE)
            
        Returns:
            Dictionary containing sentiment predictions for different time intervals
        """
        try:
            # Fetch commodity data
            commodity = self.db_session.query(Commodity).filter(Commodity.id == commodity_id).first()
            if not commodity:
                raise ValueError(f"Commodity with ID {commodity_id} not found")
            
            # Fetch required data for analysis
            market_data = await self.data_service.get_commodity_market_data(commodity.symbol)
            competition_data = await self.data_service.get_competition_data(commodity.category)
            supply_data = await self.data_service.get_supply_data(commodity.symbol)
            demand_data = await self.data_service.get_demand_data(commodity.symbol)
            news_data = await self.data_service.get_commodity_news(commodity.name)
            exporting_countries = await self.data_service.get_exporting_countries(commodity.symbol)
            
            # Generate sentiment for each time interval
            hourly_sentiment = await self._analyze_hourly_sentiment(
                commodity, market_data, competition_data, supply_data, demand_data, news_data, exporting_countries, target_country
            )
            
            daily_sentiment = await self._analyze_daily_sentiment(
                commodity, market_data, competition_data, supply_data, demand_data, news_data, exporting_countries, target_country
            )
            
            weekly_sentiment = await self._analyze_weekly_sentiment(
                commodity, market_data, competition_data, supply_data, demand_data, news_data, exporting_countries, target_country
            )
            
            monthly_sentiment = await self._analyze_monthly_sentiment(
                commodity, market_data, competition_data, supply_data, demand_data, news_data, exporting_countries, target_country
            )
            
            # Store predictions in database
            self._store_predictions(commodity_id, hourly_sentiment, daily_sentiment, weekly_sentiment, monthly_sentiment)
            
            # Prepare response
            sentiment_response = {
                "commodity_id": commodity_id,
                "commodity_name": commodity.name,
                "target_country": target_country,
                "timestamp": datetime.utcnow().isoformat(),
                "predictions": {
                    "hourly": hourly_sentiment,
                    "daily": daily_sentiment,
                    "weekly": weekly_sentiment,
                    "monthly": monthly_sentiment
                },
                "confidence_scores": {
                    "hourly": self._calculate_confidence(hourly_sentiment, market_data, news_data),
                    "daily": self._calculate_confidence(daily_sentiment, market_data, news_data),
                    "weekly": self._calculate_confidence(weekly_sentiment, market_data, news_data),
                    "monthly": self._calculate_confidence(monthly_sentiment, market_data, news_data)
                }
            }
            
            return sentiment_response
            
        except Exception as e:
            logger.error(f"Error generating sentiment prediction: {str(e)}")
            raise
    
    async def _analyze_hourly_sentiment(
        self, 
        commodity: Commodity,
        market_data: Dict,
        competition_data: Dict,
        supply_data: Dict,
        demand_data: Dict,
        news_data: List[Dict],
        exporting_countries: List[Dict],
        target_country: str
    ) -> Dict:
        """
        Analyze sentiment for hourly price movements.
        
        Returns:
            Dictionary with sentiment prediction details for hourly interval
        """
        # Extract recent price movement
        recent_prices = market_data.get("hourly_prices", [])[-6:]  # Last 6 hours
        price_trend = self._calculate_trend(recent_prices)
        
        # Analyze recent news for immediate impact
        news_sentiment = self._analyze_news_sentiment(news_data, hours=6)
        
        # Check for immediate supply disruptions
        supply_issues = self._check_immediate_supply_issues(supply_data, exporting_countries)
        
        # Analyze intraday trading patterns
        intraday_pattern = self._analyze_intraday_pattern(market_data.get("hourly_prices", []))
        
        # Calculate geographical factors impact for immediate shipping
        geo_impact = self._calculate_geo_impact(exporting_countries, target_country, timeframe="hourly")
        
        # Calculate competition impact on short-term pricing
        competition_impact = self._calculate_competition_impact(competition_data, timeframe="hourly")
        
        # Short-term demand fluctuations
        demand_impact = self._analyze_short_term_demand(demand_data)
        
        # Combine factors with appropriate weights
        sentiment_score = (
            price_trend * 0.35 +
            news_sentiment * 0.25 +
            supply_issues * 0.15 +
            intraday_pattern * 0.10 +
            geo_impact * 0.05 +
            competition_impact * 0.05 +
            demand_impact * 0.05
        )
        
        # Determine direction and strength of sentiment
        sentiment_direction = "bullish" if sentiment_score > 0.1 else "bearish" if sentiment_score < -0.1 else "neutral"
        sentiment_strength = abs(sentiment_score)
        
        return {
            "direction": sentiment_direction,
            "strength": min(sentiment_strength, 1.0),  # Cap at 1.0
            "score": sentiment_score,
            "factors": {
                "price_trend": price_trend,
                "news_sentiment": news_sentiment,
                "supply_issues": supply_issues,
                "intraday_pattern": intraday_pattern,
                "geographical_impact": geo_impact,
                "competition_impact": competition_impact,
                "demand_impact": demand_impact
            },
            "predicted_at": datetime.utcnow().isoformat()
        }
    
    async def _analyze_daily_sentiment(
        self, 
        commodity: Commodity,
        market_data: Dict,
        competition_data: Dict,
        supply_data: Dict,
        demand_data: Dict,
        news_data: List[Dict],
        exporting_countries: List[Dict],
        target_country: str
    ) -> Dict:
        """
        Analyze sentiment for daily price movements.
        
        Returns:
            Dictionary with sentiment prediction details for daily interval
        """
        # Extract recent daily price movements
        daily_prices = market_data.get("daily_prices", [])[-14:]  # Last 14 days
        price_trend = self._calculate_trend(daily_prices)
        
        # Analyze recent news with daily impact
        news_sentiment = self._analyze_news_sentiment(news_data, days=3)
        
        # Analyze daily supply changes
        supply_impact = self._analyze_supply_impact(supply_data, timeframe="daily")
        
        # Calculate daily demand patterns
        demand_impact = self._analyze_demand_impact(demand_data, timeframe="daily")
        
        # Calculate geographical factors for daily shipping
        geo_impact = self._calculate_geo_impact(exporting_countries, target_country, timeframe="daily")
        
        # Calculate competition impact
        competition_impact = self._calculate_competition_impact(competition_data, timeframe="daily")
        
        # Technical indicators for daily timeframe
        technical_indicators = self._calculate_technical_indicators(daily_prices)
        
        # Combine factors with appropriate weights
        sentiment_score = (
            price_trend * 0.25 +
            news_sentiment * 0.20 +
            supply_impact * 0.15 +
            demand_impact * 0.15 +
            geo_impact * 0.10 +
            competition_impact * 0.10 +
            technical_indicators * 0.05
        )
        
        # Determine direction and strength of sentiment
        sentiment_direction = "bullish" if sentiment_score > 0.1 else "bearish" if sentiment_score < -0.1 else "neutral"
        sentiment_strength = abs(sentiment_score)
        
        return {
            "direction": sentiment_direction,
            "strength": min(sentiment_strength, 1.0),  # Cap at 1.0
            "score": sentiment_score,
            "factors": {
                "price_trend": price_trend,
                "news_sentiment": news_sentiment,
                "supply_impact": supply_impact,
                "demand_impact": demand_impact,
                "geographical_impact": geo_impact,
                "competition_impact": competition_impact,
                "technical_indicators": technical_indicators
            },
            "predicted_at": datetime.utcnow().isoformat()
        }
    
    async def _analyze_weekly_sentiment(
        self, 
        commodity: Commodity,
        market_data: Dict,
        competition_data: Dict,
        supply_data: Dict,
        demand_data: Dict,
        news_data: List[Dict],
        exporting_countries: List[Dict],
        target_country: str
    ) -> Dict:
        """
        Analyze sentiment for weekly price movements.
        
        Returns:
            Dictionary with sentiment prediction details for weekly interval
        """
        # Extract weekly price data
        weekly_prices = market_data.get("weekly_prices", [])[-8:]  # Last 8 weeks
        price_trend = self._calculate_trend(weekly_prices)
        
        # Analyze medium-term news impact
        news_sentiment = self._analyze_news_sentiment(news_data, weeks=2)
        
        # Analyze weekly supply patterns
        supply_impact = self._analyze_supply_impact(supply_data, timeframe="weekly")
        
        # Analyze weekly demand patterns
        demand_impact = self._analyze_demand_impact(demand_data, timeframe="weekly")
        
        # Calculate geographical and political factors for weekly timeframe
        geo_impact = self._calculate_geo_impact(exporting_countries, target_country, timeframe="weekly")
        
        # Calculate competition impact on weekly pricing
        competition_impact = self._calculate_competition_impact(competition_data, timeframe="weekly")
        
        # Seasonal factors that might influence weekly pricing
        seasonal_impact = self._analyze_seasonal_factors(commodity, current_week=datetime.now().isocalendar()[1])
        
        # Combine factors with appropriate weights
        sentiment_score = (
            price_trend * 0.20 +
            news_sentiment * 0.15 +
            supply_impact * 0.15 +
            demand_impact * 0.15 +
            geo_impact * 0.15 +
            competition_impact * 0.10 +
            seasonal_impact * 0.10
        )
        
        # Determine direction and strength of sentiment
        sentiment_direction = "bullish" if sentiment_score > 0.1 else "bearish" if sentiment_score < -0.1 else "neutral"
        sentiment_strength = abs(sentiment_score)
        
        return {
            "direction": sentiment_direction,
            "strength": min(sentiment_strength, 1.0),  # Cap at 1.0
            "score": sentiment_score,
            "factors": {
                "price_trend": price_trend,
                "news_sentiment": news_sentiment,
                "supply_impact": supply_impact,
                "demand_impact": demand_impact,
                "geographical_impact": geo_impact,
                "competition_impact": competition_impact,
                "seasonal_impact": seasonal_impact
            },
            "predicted_at": datetime.utcnow().isoformat()
        }
    
    async def _analyze_monthly_sentiment(
        self, 
        commodity: Commodity,
        market_data: Dict,
        competition_data: Dict,
        supply_data: Dict,
        demand_data: Dict,
        news_data: List[Dict],
        exporting_countries: List[Dict],
        target_country: str
    ) -> Dict:
        """
        Analyze sentiment for monthly price movements.
        
        Returns:
            Dictionary with sentiment prediction details for monthly interval
        """
        # Extract monthly price data
        monthly_prices = market_data.get("monthly_prices", [])[-12:]  # Last 12 months
        price_trend = self._calculate_trend(monthly_prices)
        
        # Long-term news analysis
        news_sentiment = self._analyze_news_sentiment(news_data, months=2)
        
        # Analyze long-term supply dynamics
        supply_impact = self._analyze_supply_impact(supply_data, timeframe="monthly")
        
        # Analyze long-term demand patterns
        demand_impact = self._analyze_demand_impact(demand_data, timeframe="monthly")
        
        # Calculate geopolitical impact on monthly timeframe
        geo_impact = self._calculate_geo_impact(exporting_countries, target_country, timeframe="monthly")
        
        # Calculate long-term competitive dynamics
        competition_impact = self._calculate_competition_impact(competition_data, timeframe="monthly")
        
        # Analyze seasonal and cyclical patterns
        seasonal_impact = self._analyze_seasonal_factors(commodity, current_month=datetime.now().month)
        
        # Analyze macroeconomic indicators
        macro_impact = self._analyze_macroeconomic_indicators(commodity.category)
        
        # Combine factors with appropriate weights
        sentiment_score = (
            price_trend * 0.15 +
            news_sentiment * 0.10 +
            supply_impact * 0.15 +
            demand_impact * 0.15 +
            geo_impact * 0.15 +
            competition_impact * 0.10 +
            seasonal_impact * 0.10 +
            macro_impact * 0.10
        )
        
        # Determine direction and strength of sentiment
        sentiment_direction = "bullish" if sentiment_score > 0.1 else "bearish" if sentiment_score < -0.1 else "neutral"
        sentiment_strength = abs(sentiment_score)
        
        return {
            "direction": sentiment_direction,
            "strength": min(sentiment_strength, 1.0),  # Cap at 1.0
            "score": sentiment_score,
            "factors": {
                "price_trend": price_trend,
                "news_sentiment": news_sentiment,
                "supply_impact": supply_impact,
                "demand_impact": demand_impact,
                "geographical_impact": geo_impact,
                "competition_impact": competition_impact,
                "seasonal_impact": seasonal_impact,
                "macroeconomic_impact": macro_impact
            },
            "predicted_at": datetime.utcnow().isoformat()
        }
    
    def _calculate_trend(self, prices: List[float]) -> float:
        """Calculate the price trend from a list of prices."""
        if not prices or len(prices) < 2:
            return 0.0
        
        # Convert to numpy array for calculations
        prices_array = np.array(prices)
        
        # Calculate percentage changes
        pct_changes = np.diff(prices_array) / prices_array[:-1]
        
        # Calculate weighted average of changes (more recent changes have higher weight)
        weights = np.linspace(0.5, 1.0, len(pct_changes))
        weighted_changes = pct_changes * weights
        
        # Normalize the trend to a range between -1 and 1
        trend = np.sum(weighted_changes) / len(weighted_changes)
        normalized_trend = np.tanh(trend * 10)  # Use hyperbolic tangent to normalize
        
        return float(normalized_trend)
    
    def _analyze_news_sentiment(self, news_data: List[Dict], hours: int = None, days: int = None, weeks: int = None, months: int = None) -> float:
        """
        Analyze news sentiment for a specific timeframe.
        
        Args:
            news_data: List of news articles with sentiment scores
            hours/days/weeks/months: Timeframe to analyze
            
        Returns:
            Normalized sentiment score between -1 and 1
        """
        if not news_data:
            return 0.0
        
        # Determine cutoff time
        now = datetime.utcnow()
        if hours:
            cutoff_time = now - timedelta(hours=hours)
        elif days:
            cutoff_time = now - timedelta(days=days)
        elif weeks:
            cutoff_time = now - timedelta(weeks=weeks)
        elif months:
            cutoff_time = now - timedelta(days=30*months)
        else:
            cutoff_time = now - timedelta(days=1)  # Default to 1 day
            
        # Filter news by date
        filtered_news = [
            news for news in news_data 
            if datetime.fromisoformat(news.get("published_at", now.isoformat())) >= cutoff_time
        ]
        
        if not filtered_news:
            return 0.0
            
        # Extract sentiment scores and apply recency weighting
        sentiments = []
        weights = []
        
        for i, news in enumerate(filtered_news):
            news_time = datetime.fromisoformat(news.get("published_at", now.isoformat()))
            time_diff = (now - news_time).total_seconds()
            
            # Calculate recency weight (more recent = higher weight)
            recency_weight = 1.0 / (1.0 + time_diff / 86400)  # 86400 seconds in a day
            
            # Calculate importance weight based on source reliability and reach
            importance_weight = news.get("source_reliability", 0.5) * news.get("source_reach", 0.5)
            
            # Calculate final weight
            weight = recency_weight * importance_weight
            
            sentiments.append(news.get("sentiment_score", 0.0))
            weights.append(weight)
            
        # Calculate weighted average sentiment
        if sum(weights) > 0:
            weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
        else:
            weighted_sentiment = 0.0
            
        return weighted_sentiment
    
    def _check_immediate_supply_issues(self, supply_data: Dict, exporting_countries: List[Dict]) -> float:
        """
        Check for immediate supply disruptions that could impact hourly pricing.
        
        Returns:
            Score between -1 (severe supply disruption) and 1 (supply surplus)
        """
        if not supply_data or not exporting_countries:
            return 0.0
            
        # Check for reported disruptions
        disruption_score = supply_data.get("immediate_disruption_score", 0.0)
        
        # Check shipping delays from major exporters
        shipping_delays = 0.0
        for country in exporting_countries:
            delay_factor = country.get("shipping_delay_factor", 0.0)
            country_weight = country.get("export_volume_percentage", 0.0) / 100
            shipping_delays += delay_factor * country_weight
            
        # Combine disruption score and shipping delays
        combined_score = -0.7 * disruption_score - 0.3 * shipping_delays
        
        return max(-1.0, min(1.0, combined_score))  # Normalize between -1 and 1
    
    def _analyze_intraday_pattern(self, hourly_prices: List[float]) -> float:
        """
        Analyze intraday trading patterns to predict short-term price movement.
        
        Returns:
            Score between -1 (bearish) and 1 (bullish)
        """
        if not hourly_prices or len(hourly_prices) < 24:
            return 0.0
            
        # Get the last 24 hours
        last_24_hours = hourly_prices[-24:]
        
        # Calculate volatility
        returns = np.diff(last_24_hours) / last_24_hours[:-1]
        volatility = np.std(returns)
        
        # Calculate momentum (last 6 hours vs previous 6 hours)
        recent_6h = np.mean(last_24_hours[-6:])
        previous_6h = np.mean(last_24_hours[-12:-6])
        momentum = (recent_6h - previous_6h) / previous_6h
        
        # Identify pattern formation (simplified)
        pattern_score = 0.0
        
        # Rising trend in last few hours
        if all(last_24_hours[-4:-1][i] < last_24_hours[-4:-1][i+1] for i in range(2)):
            pattern_score += 0.3
            
        # Falling trend in last few hours
        if all(last_24_hours[-4:-1][i] > last_24_hours[-4:-1][i+1] for i in range(2)):
            pattern_score -= 0.3
            
        # Combine factors
        intraday_score = 0.5 * np.tanh(momentum * 5) + 0.3 * pattern_score - 0.2 * min(volatility * 10, 1.0)
        
        return max(-1.0, min(1.0, intraday_score))
    
    def _calculate_geo_impact(self, exporting_countries: List[Dict], target_country: str, timeframe: str) -> float:
        """
        Calculate the impact of geographical factors on pricing.
        
        Args:
            exporting_countries: List of main exporting countries
            target_country: Target importing country
            timeframe: Analysis timeframe (hourly, daily, weekly, monthly)
            
        Returns:
            Impact score between -1 (negative impact) and 1 (positive impact)
        """
        if not exporting_countries:
            return 0.0
            
        total_impact = 0.0
        total_weight = 0.0
        
        for country in exporting_countries:
            country_name = country.get("country_name", "")
            export_percentage = country.get("export_volume_percentage", 0.0) / 100
            
            # Calculate shipping impact (time, cost, reliability)
            shipping_impact = calculate_shipping_impact(country_name, target_country)
            
            # Calculate political stability impact
            political_stability = country.get("political_stability", 0.5)
            
            # Calculate trade relations impact
            trade_relations = country.get("trade_relations_with_" + target_country.lower(), 0.5)
            
            # Apply different weights based on timeframe
            if timeframe == "hourly":
                # For hourly, shipping disruptions matter most
                impact = 0.8 * shipping_impact + 0.1 * political_stability + 0.1 * trade_relations
            elif timeframe == "daily":
                # For daily, balance shipping and political factors
                impact = 0.6 * shipping_impact + 0.2 * political_stability + 0.2 * trade_relations
            elif timeframe == "weekly":
                # For weekly, political factors become more important
                impact = 0.4 * shipping_impact + 0.3 * political_stability + 0.3 * trade_relations
            else:  # monthly
                # For monthly, political and trade relations are most important
                impact = 0.2 * shipping_impact + 0.4 * political_stability + 0.4 * trade_relations
                
            # Add to weighted average
            total_impact += impact * export_percentage
            total_weight += export_percentage
            
        if total_weight > 0:
            return total_impact / total_weight
        else:
            return 0.0
    
    def _calculate_competition_impact(self, competition_data: Dict, timeframe: str) -> float:
        """
        Calculate the impact of competition on pricing.
        
        Args:
            competition_data: Data about competing products and suppliers
            timeframe: Analysis timeframe
            
        Returns:
            Impact score between -1 (negative impact) and 1 (positive impact)
        """
        if not competition_data:
            return 0.0
            
        # Extract competition metrics
        market_concentration = competition_data.get("market_concentration", 0.5)  # Higher means less competition
        price_elasticity = competition_data.get("price_elasticity", 0.5)  # Higher means more sensitive to price
        substitute_availability = competition_data.get("substitute_availability", 0.5)  # Higher means more substitutes
        
        # Calculate base competition score
        # Low concentration, high elasticity, and high substitute availability
        # all contribute to higher competition (negative impact on prices)
        base_competition_score = (market_concentration - 0.5) - (price_elasticity - 0.5) - (substitute_availability - 0.5)
        
        # Apply timeframe-specific adjustments
        if timeframe == "hourly":
            # Short-term competition has less impact
            adjusted_score = base_competition_score * 0.3
        elif timeframe == "daily":
            # Daily competition has moderate impact
            adjusted_score = base_competition_score * 0.5
        elif timeframe == "weekly":
            # Weekly competition has significant impact
            adjusted_score = base_competition_score * 0.8
        else:  # monthly
            # Monthly competition has full impact
            adjusted_score = base_competition_score
            
        # Normalize to range between -1 and 1
        return max(-1.0, min(1.0, adjusted_score))
    
    def _analyze_short_term_demand(self, demand_data: Dict) -> float:
        """
        Analyze short-term demand fluctuations for hourly predictions.
        
        Returns:
            Score between -1 (falling demand) and 1 (rising demand)
        """
        if not demand_data:
            return 0.0
            
        # Extract hourly demand indicators
        hourly_change = demand_data.get("hourly_demand_change", 0.0)
        intraday_pattern = demand_data.get("intraday_demand_pattern", 0.0)
        
        # Combine indicators
        demand_score = 0.7 * hourly_change + 0.3 * intraday_pattern
        
        return max(-1.0, min(1.0, demand_score))
    
    def _analyze_supply_impact(self, supply_data: Dict, timeframe: str) -> float:
        """
        Analyze supply dynamics for the specified timeframe.
        
        Returns:
            Score between -1 (supply shortage) and 1 (supply surplus)
        """
        if not supply_data:
            return 0.0
            
        # Extract supply metrics
        current_supply_level = supply_data.get("current_supply_level", 0.5)  # 0.5 means balanced
        supply_trend = supply_data.get("supply_trend", 0.0)  # Positive means increasing
        planned_production_changes = supply_data.get("planned_production_changes", 0.0)
        
        # Apply different weights based on timeframe
        if timeframe == "daily":
            impact = 0.7 * (current_supply_level - 0.5) + 0.3 * supply_trend
        elif timeframe == "weekly":
            impact = 0.5 * (current_supply_level - 0.5) + 0.3 * supply_trend + 0.2 * planned_production_changes
        else:  # monthly
            impact = 0.3 * (current_supply_level - 0.5) + 0.3 * supply_trend + 0.4 * planned_production_changes
            
        return max(-1.0, min(1.0, impact))
    
    def _analyze_demand_impact(self, demand_data: Dict, timeframe: str) -> float:
        """
        Analyze demand patterns for the specified timeframe.
        
        Returns:
            Score between -1 (falling demand) and 1 (rising demand)
        """
        if not demand_data:
            return 0.0
            
        # Extract demand metrics
        current_demand_level = demand_data.get("current_demand_level", 0.5)  # 0.5 means balanced
        demand_trend = demand_data.get("demand_trend", 0.0)  # Positive means increasing
        forecast_demand_change = demand_data.get("forecast_demand_change", 0.0)
        
        # Apply different weights based on timeframe
        if timeframe == "daily":
            impact = 0.7 * (current_demand_level - 0.5) + 0.3 * demand_trend
        elif timeframe == "weekly":
            impact = 0.5 * (current_demand_level - 0.5) + 0.3 * demand_trend + 0.2 * forecast_demand_change
        else:  # monthly
            impact = 0.3 * (current_demand_level - 0.5) + 0.3 * demand_trend + 0.4 * forecast_demand_change
            
        return max(-1.0, min(1.0, impact))
    
    def _calculate_technical_indicators(self, prices: List[float]) -> float:
        """
        Calculate technical indicators for price prediction.
        
        Returns:
            Score between -1 (bearish) and 1 (bullish)
        """
        if not prices or len(prices) < 20:
            return 0.0
            
        prices_array = np.array(prices)
        
        # Calculate simple moving averages
        sma_5 = np.mean(prices_array[-5:])
        sma_20 = np.mean(prices_array[-20:])
        
        # Calculate relative strength index (simplified)
        diff = np.diff(prices_array)
        gains = diff.copy()
        losses = diff.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD (simplified)
        ema_12 = prices_array[-1]  # This is a simplified EMA calculation
        ema_26 = np.mean(prices_array[-26:])
        macd = ema_12 - ema_26
        
        # Combine indicators
        sma_signal = 1 if sma_5 > sma_20 else -1 if sma_5 < sma_20 else 0
        rsi_signal = 1 if rsi < 30 else -1 if rsi > 70 else 0
        macd_signal = 1 if macd > 0 else -1 if macd < 0 else 0
        
        # Weight and combine signals
        technical_score = 0.4 * sma_signal + 0.3 * rsi_signal + 0.3 * macd_signal
        
        return technical_score
    
    def _analyze_seasonal_factors(self, commodity: Commodity, current_week: int = None, current_month: int = None) -> float:
        """
        Analyze seasonal and cyclical patterns affecting commodity prices.
        
        Args:
            commodity: Commodity object
            current_week: Current week number (1-52)
            current_month: Current month number (1-12)
            
        Returns:
            Score between -1 (bearish seasonal impact) and 1 (bullish seasonal impact)
        """
        # Get commodity category for seasonal analysis
        category = commodity.category
        
        # Define seasonal patterns for different commodity categories
        seasonal_patterns = {
            "agriculture": {
                "monthly": [0.2, 0.3, 0.1, -0.1, -0.3, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],  # Monthly pattern
                "weekly": [0.1 * (i % 4 - 2) for i in range(52)]  # Weekly pattern (simplified)
            },
            "energy": {
                "monthly": [0.5, 0.3, 0.0, -0.2, -0.3, 0.0, 0.2, 0.3, 0.1, 0.0, 0.1, 0.4],
                "weekly": [0.05 * ((i % 12) - 6) for i in range(52)]
            },
            "metals": {
                "monthly": [0.1, 0.1, 0.0, 0.0, -0.1, -0.1, -0.1, 0.0, 0.1, 0.2, 0.1, 0.0],
                "weekly": [0.02 * ((i % 8) - 4) for i in range(52)]
            },
            "livestock": {
                "monthly": [0.0, 0.1, 0.2, 0.3, 0.2, 0.0, -0.2, -0.3, -0.2, -0.1, 0.0, 0.1],
                "weekly": [0.08 * math.sin(2 * math.pi * i / 52) for i in range(52)]
            }
        }
        
        # Get default patterns if category not found
        category_patterns = seasonal_patterns.get(category.lower(), {
            "monthly": [0.0] * 12,
            "weekly": [0.0] * 52
        })
        
        # Determine seasonal impact based on current time
        if current_month is not None:
            # Use monthly patterns
            month_idx = current_month - 1  # Convert to 0-indexed
            return category_patterns["monthly"][month_idx]
        elif current_week is not None:
            # Use weekly patterns
            week_idx = current_week - 1  # Convert to 0-indexed
            return category_patterns["weekly"][week_idx]
        else:
            # Default to current month
            current_month = datetime.now().month
            month_idx = current_month - 1
            return category_patterns["monthly"][month_idx]
    
    def _analyze_macroeconomic_indicators(self, commodity_category: str) -> float:
        """
        Analyze macroeconomic indicators affecting long-term commodity prices.
        
        Args:
            commodity_category: Category of the commodity
            
        Returns:
            Score between -1 (bearish macro impact) and 1 (bullish macro impact)
        """
        try:
            # Fetch macroeconomic data from data service
            macro_data = self.data_service.get_macroeconomic_indicators_sync()
            
            if not macro_data:
                return 0.0
                
            # Extract key indicators
            inflation_rate = macro_data.get("inflation_rate", 2.0)  # Percent
            interest_rate = macro_data.get("interest_rate", 2.0)  # Percent
            gdp_growth = macro_data.get("gdp_growth", 2.0)  # Percent
            currency_strength = macro_data.get("usd_index", 100.0)  # USD index
            oil_price = macro_data.get("oil_price", 70.0)  # USD per barrel
            
            # Define category-specific sensitivities to macro factors
            sensitivities = {
                "agriculture": {
                    "inflation": 0.3,
                    "interest_rate": -0.2,
                    "gdp_growth": 0.2,
                    "currency_strength": -0.4,
                    "oil_price": -0.3
                },
                "energy": {
                    "inflation": 0.4,
                    "interest_rate": -0.3,
                    "gdp_growth": 0.5,
                    "currency_strength": -0.3,
                    "oil_price": 0.8
                },
                "metals": {
                    "inflation": 0.6,
                    "interest_rate": -0.4,
                    "gdp_growth": 0.4,
                    "currency_strength": -0.5,
                    "oil_price": 0.2
                },
                "livestock": {
                    "inflation": 0.3,
                    "interest_rate": -0.1,
                    "gdp_growth": 0.3,
                    "currency_strength": -0.3,
                    "oil_price": -0.2
                }
            }
            
            # Get default sensitivities if category not found
            category_sensitivities = sensitivities.get(commodity_category.lower(), {
                "inflation": 0.3,
                "interest_rate": -0.2,
                "gdp_growth": 0.3,
                "currency_strength": -0.3,
                "oil_price": 0.0
            })
            
            # Calculate inflation impact (higher inflation generally increases commodity prices)
            inflation_impact = (inflation_rate - 2.0) / 10.0  # Normalize around 2% target
            inflation_impact *= category_sensitivities["inflation"]
            
            # Calculate interest rate impact (higher rates generally decrease commodity prices)
            interest_impact = (interest_rate - 2.5) / 5.0  # Normalize around 2.5% neutral rate
            interest_impact *= category_sensitivities["interest_rate"]
            
            # Calculate GDP growth impact (higher growth generally increases commodity prices)
            gdp_impact = (gdp_growth - 2.0) / 5.0  # Normalize around 2% average growth
            gdp_impact *= category_sensitivities["gdp_growth"]
            
            # Calculate currency strength impact (stronger USD generally decreases commodity prices)
            currency_impact = (currency_strength - 100.0) / 20.0  # Normalize around index 100
            currency_impact *= category_sensitivities["currency_strength"]
            
            # Calculate oil price impact (varies by commodity category)
            oil_impact = (oil_price - 70.0) / 30.0  # Normalize around $70/barrel
            oil_impact *= category_sensitivities["oil_price"]
            
            # Combine all impacts
            macro_score = (
                inflation_impact +
                interest_impact +
                gdp_impact +
                currency_impact +
                oil_impact
            )
            
            # Normalize to range between -1 and 1
            macro_score = max(-1.0, min(1.0, macro_score))
            
            return macro_score
            
        except Exception as e:
            logger.error(f"Error analyzing macroeconomic indicators: {str(e)}")
            return 0.0
    
    def _calculate_confidence(self, sentiment: Dict, market_data: Dict, news_data: List[Dict]) -> float:
        """
        Calculate confidence level for the sentiment prediction.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        base_confidence = 0.7
        
        # Adjustments based on data quality and quantity
        
        # Market data quality adjustment
        data_points = len(market_data.get("hourly_prices", [])) + len(market_data.get("daily_prices", []))
        if data_points < 10:
            market_data_adjustment = -0.2
        elif data_points < 30:
            market_data_adjustment = -0.1
        elif data_points < 60:
            market_data_adjustment = 0.0
        else:
            market_data_adjustment = 0.1
            
        # News data quality adjustment
        news_count = len(news_data)
        if news_count < 5:
            news_data_adjustment = -0.15
        elif news_count < 15:
            news_data_adjustment = -0.05
        elif news_count < 30:
            news_data_adjustment = 0.0
        else:
            news_data_adjustment = 0.05
            
        # Sentiment strength adjustment (stronger sentiments have higher confidence)
        sentiment_strength = abs(sentiment.get("score", 0.0))
        if sentiment_strength < 0.2:
            strength_adjustment = -0.1  # Very weak signal
        elif sentiment_strength < 0.4:
            strength_adjustment = -0.05  # Weak signal
        elif sentiment_strength < 0.6:
            strength_adjustment = 0.0  # Moderate signal
        elif sentiment_strength < 0.8:
            strength_adjustment = 0.05  # Strong signal
        else:
            strength_adjustment = 0.1  # Very strong signal
            
        # Calculate final confidence
        confidence = base_confidence + market_data_adjustment + news_data_adjustment + strength_adjustment
        
        # Ensure confidence is within valid range
        confidence = max(0.1, min(0.95, confidence))
        
        return confidence
    
    def _store_predictions(self, commodity_id: int, hourly: Dict, daily: Dict, weekly: Dict, monthly: Dict) -> None:
        """
        Store sentiment predictions in the database.
        
        Args:
            commodity_id: ID of the commodity
            hourly/daily/weekly/monthly: Sentiment prediction dictionaries
        """
        try:
            # Create prediction records
            timestamp = datetime.utcnow()
            
            hourly_prediction = SentimentPrediction(
                commodity_id=commodity_id,
                timeframe="hourly",
                direction=hourly["direction"],
                strength=hourly["strength"],
                score=hourly["score"],
                factors=json.dumps(hourly["factors"]),
                confidence=self._calculate_confidence(hourly, {}, []),
                created_at=timestamp
            )
            
            daily_prediction = SentimentPrediction(
                commodity_id=commodity_id,
                timeframe="daily",
                direction=daily["direction"],
                strength=daily["strength"],
                score=daily["score"],
                factors=json.dumps(daily["factors"]),
                confidence=self._calculate_confidence(daily, {}, []),
                created_at=timestamp
            )
            
            weekly_prediction = SentimentPrediction(
                commodity_id=commodity_id,
                timeframe="weekly",
                direction=weekly["direction"],
                strength=weekly["strength"],
                score=weekly["score"],
                factors=json.dumps(weekly["factors"]),
                confidence=self._calculate_confidence(weekly, {}, []),
                created_at=timestamp
            )
            
            monthly_prediction = SentimentPrediction(
                commodity_id=commodity_id,
                timeframe="monthly",
                direction=monthly["direction"],
                strength=monthly["strength"],
                score=monthly["score"],
                factors=json.dumps(monthly["factors"]),
                confidence=self._calculate_confidence(monthly, {}, []),
                created_at=timestamp
            )
            
            # Add to database
            self.db_session.add(hourly_prediction)
            self.db_session.add(daily_prediction)
            self.db_session.add(weekly_prediction)
            self.db_session.add(monthly_prediction)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Error storing predictions: {str(e)}")
            self.db_session.rollback()
            
    async def get_sentiment_history(self, commodity_id: int, timeframe: str, days: int = 30) -> List[Dict]:
        """
        Get historical sentiment predictions for a commodity.
        
        Args:
            commodity_id: ID of the commodity
            timeframe: Timeframe for analysis (hourly, daily, weekly, monthly)
            days: Number of days of history to retrieve
            
        Returns:
            List of historical sentiment predictions
        """
        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Query database for historical predictions
            predictions = self.db_session.query(SentimentPrediction).filter(
                SentimentPrediction.commodity_id == commodity_id,
                SentimentPrediction.timeframe == timeframe,
                SentimentPrediction.created_at >= cutoff_date
            ).order_by(SentimentPrediction.created_at.asc()).all()
            
            # Format results
            result = []
            for prediction in predictions:
                result.append({
                    "id": prediction.id,
                    "commodity_id": prediction.commodity_id,
                    "timeframe": prediction.timeframe,
                    "direction": prediction.direction,
                    "strength": prediction.strength,
                    "score": prediction.score,
                    "factors": json.loads(prediction.factors),
                    "confidence": prediction.confidence,
                    "created_at": prediction.created_at.isoformat()
                })
                
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving sentiment history: {str(e)}")
            return []
    
    async def get_sentiment_accuracy(self, commodity_id: int, timeframe: str, days: int = 30) -> Dict:
        """
        Calculate accuracy of past sentiment predictions.
        
        Args:
            commodity_id: ID of the commodity
            timeframe: Timeframe for analysis
            days: Number of days of history to analyze
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            # Get historical predictions
            predictions = await self.get_sentiment_history(commodity_id, timeframe, days)
            
            if not predictions:
                return {
                    "accuracy": 0.0,
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "incorrect_predictions": 0,
                    "pending_predictions": 0,
                    "analysis_period_days": days
                }
                
            # Get actual price data for comparison
            commodity = self.db_session.query(Commodity).filter(Commodity.id == commodity_id).first()
            if not commodity:
                raise ValueError(f"Commodity with ID {commodity_id} not found")
                
            market_data = await self.data_service.get_commodity_market_data(commodity.symbol)
            
            # Calculate verification window based on timeframe
            if timeframe == "hourly":
                verification_hours = 1
            elif timeframe == "daily":
                verification_hours = 24
            elif timeframe == "weekly":
                verification_hours = 24 * 7
            else:  # monthly
                verification_hours = 24 * 30
                
            # Calculate accuracy
            total_predictions = len(predictions)
            correct_predictions = 0
            incorrect_predictions = 0
            pending_predictions = 0
            
            for prediction in predictions:
                # Skip predictions that haven't had time to be verified
                prediction_time = datetime.fromisoformat(prediction["created_at"])
                verification_time = prediction_time + timedelta(hours=verification_hours)
                
                if verification_time > datetime.utcnow():
                    pending_predictions += 1
                    continue
                    
                # Get actual price movement
                actual_movement = self._get_actual_price_movement(
                    market_data, prediction_time, verification_hours
                )
                
                # Compare prediction with actual movement
                if (prediction["direction"] == "bullish" and actual_movement > 0) or \
                   (prediction["direction"] == "bearish" and actual_movement < 0) or \
                   (prediction["direction"] == "neutral" and abs(actual_movement) < 0.01):
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
                    
            # Calculate accuracy percentage
            verified_predictions = correct_predictions + incorrect_predictions
            accuracy = (correct_predictions / verified_predictions) * 100 if verified_predictions > 0 else 0
            
            return {
                "accuracy": round(accuracy, 2),
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "incorrect_predictions": incorrect_predictions,
                "pending_predictions": pending_predictions,
                "analysis_period_days": days
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment accuracy: {str(e)}")
            return {
                "accuracy": 0.0,
                "error": str(e)
            }
    
    def _get_actual_price_movement(self, market_data: Dict, prediction_time: datetime, verification_hours: int) -> float:
        """
        Get the actual price movement after a prediction.
        
        Args:
            market_data: Market data dictionary
            prediction_time: When the prediction was made
            verification_hours: Hours to look ahead for verification
            
        Returns:
            Percentage price movement
        """
        # Determine price key based on verification period
        if verification_hours <= 1:
            price_key = "hourly_prices"
            time_key = "hourly_timestamps"
        elif verification_hours <= 24:
            price_key = "daily_prices"
            time_key = "daily_timestamps"
        elif verification_hours <= 24*7:
            price_key = "weekly_prices"
            time_key = "weekly_timestamps"
        else:
            price_key = "monthly_prices"
            time_key = "monthly_timestamps"
            
        # Get price data and timestamps
        prices = market_data.get(price_key, [])
        timestamps = market_data.get(time_key, [])
        
        if not prices or not timestamps or len(prices) != len(timestamps):
            return 0.0
            
        # Find price at prediction time
        start_idx = None
        end_idx = None
        
        for i, ts in enumerate(timestamps):
            timestamp = datetime.fromisoformat(ts)
            if timestamp >= prediction_time and start_idx is None:
                start_idx = i
            if timestamp >= prediction_time + timedelta(hours=verification_hours) and end_idx is None:
                end_idx = i
                break
                
        # Handle edge cases
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(prices) - 1
            
        # Calculate price movement
        if start_idx >= len(prices) or end_idx >= len(prices) or start_idx == end_idx:
            return 0.0
            
        start_price = prices[start_idx]
        end_price = prices[end_idx]
        
        price_movement = (end_price - start_price) / start_price
        
        return price_movement

# Create factory function for dependency injection
def get_sentiment_service(db_session: Session = None):
    """Factory function to create a SentimentAnalysisService instance."""
    if db_session is None:
        # Import here to avoid circular imports
        from database import get_db
        db_session = next(get_db())
    return SentimentAnalysisService(db_session)
