# services/data_service.py
import requests
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import json
import random

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self):
        self.alpha_vantage_api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')
        self.finnhub_api_key = os.environ.get('FINNHUB_API_KEY', 'demo')
        self.commodities_api_key = os.environ.get('COMMODITIES_API_KEY', 'demo')
        self.base_url = 'https://www.alphavantage.co/query'
        self.finnhub_url = 'https://finnhub.io/api/v1'
        self.commodities_url = 'https://www.commodities-api.com/api'
        self.cache = {}
        self.cache_expiry = {}
        
    def _get_cache_key(self, func_name, *args):
        return f"{func_name}:{'_'.join(str(arg) for arg in args)}"
    
    def _get_from_cache(self, cache_key, max_age_seconds=3600):
        if cache_key in self.cache and cache_key in self.cache_expiry:
            if (datetime.utcnow() - self.cache_expiry[cache_key]).total_seconds() < max_age_seconds:
                return self.cache[cache_key]
        return None
    
    def _set_in_cache(self, cache_key, data):
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = datetime.utcnow()
    
    def get_price_data(self, commodity_name, days=30):
        """Get historical price data for a commodity"""
        cache_key = self._get_cache_key('get_price_data', commodity_name, days)
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Map commodity name to Alpha Vantage symbol
            symbol_map = {
                'Oil': 'WTI',
                'Gold': 'GOLD',
                'Silver': 'SILVER',
                'Copper': 'COPPER',
                'Natural Gas': 'NATURAL_GAS',
                'Wheat': 'WHEAT',
                'Coffee': 'COFFEE'
            }
            
            symbol = symbol_map.get(commodity_name, commodity_name.upper())
            
            # In a production environment, this would use the actual API:
            # params = {
            #     'function': 'TIME_SERIES_DAILY',
            #     'symbol': symbol,
            #     'apikey': self.alpha_vantage_api_key,
            #     'outputsize': 'full'
            # }
            # response = requests.get(self.base_url, params=params)
            # data = response.json()
            
            # For demo purposes, generating synthetic data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Generate synthetic price data
            dates = []
            prices = []
            current_date = start_date
            base_price = 100  # Starting price
            
            # Get commodity-specific starting price
            if commodity_name == 'Oil':
                base_price = 75.0
            elif commodity_name == 'Gold':
                base_price = 1900.0
            elif commodity_name == 'Silver':
                base_price = 25.0
            elif commodity_name == 'Copper':
                base_price = 4.0
            elif commodity_name == 'Natural Gas':
                base_price = 3.5
            elif commodity_name == 'Wheat':
                base_price = 7.0
            elif commodity_name == 'Coffee':
                base_price = 2.0
            
            # Generate daily prices with some randomness
            current_price = base_price
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Only weekdays
                    dates.append(current_date.strftime('%Y-%m-%d'))
                    
                    # Create a trend with some random noise
                    change_percent = random.normalvariate(0, 0.01)  # Mean 0, std dev 1%
                    current_price = current_price * (1 + change_percent)
                    prices.append(round(current_price, 2))
                
                current_date += timedelta(days=1)
            
            # Convert to pandas DataFrame
            df = pd.DataFrame({
                'date': dates,
                'price': prices
            })
            
            self._set_in_cache(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching price data for {commodity_name}: {str(e)}")
            # Return empty DataFrame in case of error
            return pd.DataFrame(columns=['date', 'price'])
    
    def get_competition_data(self, commodity_name):
        """Get competition data for a commodity"""
        cache_key = self._get_cache_key('get_competition_data', commodity_name)
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # In a production environment, this would use an actual API
            # For demo purposes, generating synthetic data
            
            # Competition intensity (0-1): 0 means low competition, 1 means high
            competition_map = {
                'Oil': 0.85,
                'Gold': 0.65,
                'Silver': 0.6,
                'Copper': 0.75,
                'Natural Gas': 0.8,
                'Wheat': 0.7,
                'Coffee': 0.6
            }
            
            # Number of major producers
            producers_map = {
                'Oil': ['Saudi Arabia', 'Russia', 'USA', 'Iran', 'Iraq', 'UAE', 'Kuwait', 'Venezuela'],
                'Gold': ['China', 'Australia', 'Russia', 'USA', 'Canada', 'Ghana', 'South Africa'],
                'Silver': ['Mexico', 'Peru', 'China', 'Russia', 'Poland', 'Australia', 'Chile'],
                'Copper': ['Chile', 'Peru', 'China', 'DRC', 'USA', 'Australia', 'Russia'],
                'Natural Gas': ['USA', 'Russia', 'Iran', 'Qatar', 'Canada', 'Norway', 'Australia'],
                'Wheat': ['China', 'India', 'Russia', 'USA', 'France', 'Australia', 'Canada'],
                'Coffee': ['Brazil', 'Vietnam', 'Colombia', 'Indonesia', 'Ethiopia', 'Honduras', 'India']
            }
            
            # Market share data (approximate)
            market_share = {}
            producers = producers_map.get(commodity_name, ['Unknown'])
            
            # Generate synthetic market share data
            remaining_share = 100.0
            for i, producer in enumerate(producers):
                if i == len(producers) - 1:
                    # Last producer gets remaining share
                    market_share[producer] = round(remaining_share, 1)
                else:
                    # Distribute shares with some randomness but following Pareto principle
                    share = remaining_share * (0.8 if i == 0 else 0.5 if i == 1 else 0.3) * random.uniform(0.8, 1.2)
                    share = min(share, remaining_share - 1.0)  # Ensure we leave at least 1% for the last
                    market_share[producer] = round(share, 1)
                    remaining_share -= market_share[producer]
            
            competition_data = {
                'intensity': competition_map.get(commodity_name, 0.5),
                'num_producers': len(producers),
                'producers': producers,
                'market_share': market_share
            }
            
            self._set_in_cache(cache_key, competition_data)
            return competition_data
            
        except Exception as e:
            logger.error(f"Error fetching competition data for {commodity_name}: {str(e)}")
            return {
                'intensity': 0.5,
                'num_producers': 0,
                'producers': [],
                'market_share': {}
            }
    
    def get_availability_data(self, commodity_name):
        """Get availability/supply data for a commodity"""
        cache_key = self._get_cache_key('get_availability_data', commodity_name)
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # In a production environment, this would use an actual API
            # For demo purposes, generating synthetic data
            
            # Supply levels (0-1): 0 means low supply, 1 means abundant supply
            supply_level_map = {
                'Oil': 0.7,
                'Gold': 0.5,
                'Silver': 0.6,
                'Copper': 0.65,
                'Natural Gas': 0.75,
                'Wheat': 0.8,
                'Coffee': 0.6
            }
            
            # Recent disruptions
            disruptions_map = {
                'Oil': ['Minor production issues in Nigeria', 'Maintenance in North Sea fields'],
                'Gold': ['Labor strikes in South Africa'],
                'Silver': ['Production delays in Peru'],
                'Copper': ['Environmental concerns limiting new projects'],
                'Natural Gas': ['Pipeline maintenance in Europe'],
                'Wheat': ['Drought affecting Australia harvests'],
                'Coffee': ['Frost damage in Brazil', 'Lower yields in Colombia']
            }
            
            # Reserve estimates (years)
            reserves_map = {
                'Oil': 50,
                'Gold': 20,
                'Silver': 25,
                'Copper': 40,
                'Natural Gas': 55,
                'Wheat': None,  # Renewable annually
                'Coffee': None  # Renewable annually
            }
            
            # Generate trend in production (year-over-year percentage change)
            production_trend = random.normalvariate(0.02, 0.05)  # Mean 2% growth, std dev 5%
            
            availability_data = {
                'supply_level': supply_level_map.get(commodity_name, 0.5),
                'disruptions': disruptions_map.get(commodity_name, []),
                'reserves_years': reserves_map.get(commodity_name),
                'production_trend': round(production_trend, 4)
            }
            
            self._set_in_cache(cache_key, availability_data)
            return availability_data
            
        except Exception as e:
            logger.error(f"Error fetching availability data for {commodity_name}: {str(e)}")
            return {
                'supply_level': 0.5,
                'disruptions': [],
                'reserves_years': None,
                'production_trend': 0.0
            }
    
    def get_demand_data(self, commodity_name):
        """Get demand data for a commodity"""
        cache_key = self._get_cache_key('get_demand_data', commodity_name)
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # In a production environment, this would use an actual API
            # For demo purposes, generating synthetic data
            
            # Demand growth (year-over-year percentage)
            demand_growth_map = {
                'Oil': 0.015,  # 1.5% 
                'Gold': 0.03,   # 3%
                'Silver': 0.025, # 2.5%
                'Copper': 0.04,  # 4%
                'Natural Gas': 0.03, # 3%
                'Wheat': 0.02,   # 2%
                'Coffee': 0.025  # 2.5%
            }
            
            # Primary consumers (countries or sectors)
            consumers_map = {
                'Oil': ['Transportation', 'Industry', 'Power Generation', 'Residential'],
                'Gold': ['Jewelry', 'Investment', 'Technology', 'Central Banks'],
                'Silver': ['Industry', 'Jewelry', 'Photography', 'Investment'],
                'Copper': ['Construction', 'Electronics', 'Transportation', 'Industrial Machinery'],
                'Natural Gas': ['Power Generation', 'Industry', 'Residential', 'Commercial'],
                'Wheat': ['Food Production', 'Animal Feed', 'Industrial Applications'],
                'Coffee': ['Household Consumption', 'Food Service', 'Ready-to-Drink']
            }
            
            # Seasonal factors
            seasonal_factors_map = {
                'Oil': 'Higher demand in summer driving season and winter heating',
                'Gold': 'Higher demand during uncertainty and festival seasons',
                'Silver': 'Consistent industrial demand, higher jewelry demand seasonally',
                'Copper': 'Consistent industrial demand with slight construction seasonality',
                'Natural Gas': 'Higher demand in winter months for heating',
                'Wheat': 'Demand consistent, supply varies by harvest seasons',
                'Coffee': 'Relatively consistent demand with slight seasonal variations'
            }
            
            # Recent demand shifts
            demand_shifts_map = {
                'Oil': ['Increasing EV adoption affecting gasoline demand', 'Recovering air travel post-pandemic'],
                'Gold': ['Increased investment demand due to economic uncertainty'],
                'Silver': ['Growing industrial demand for solar panels', 'Electronics manufacturing recovery'],
                'Copper': ['Strong demand for infrastructure projects', 'Growing EV production requiring more copper'],
                'Natural Gas': ['Transition from coal to natural gas for power generation', 'LNG export growth'],
                'Wheat': ['Changing dietary preferences', 'Growing population driving base demand'],
                'Coffee': ['Growth in specialty coffee market', 'Expanding coffee culture in traditionally tea markets']
            }
            
            # UAE specific demand factors
            uae_factors_map = {
                'Oil': 'Major producer, less dependent on imports',
                'Gold': 'Strong demand for investment and jewelry',
                'Silver': 'Moderate demand for investment and industrial uses',
                'Copper': 'Demand driven by construction and infrastructure development',
                'Natural Gas': 'Growing demand for power generation',
                'Wheat': 'High import dependency for food security',
                'Coffee': 'Growing urban coffee culture and tourism sector'
            }
            
            demand_data = {
                'growth_rate': demand_growth_map.get(commodity_name, 0.02),
                'primary_consumers': consumers_map.get(commodity_name, []),
                'seasonal_factors': seasonal_factors_map.get(commodity_name, 'No significant seasonality'),
                'recent_shifts': demand_shifts_map.get(commodity_name, []),
                'uae_specific': uae_factors_map.get(commodity_name, 'Standard market dynamics')
            }
            
            self._set_in_cache(cache_key, demand_data)
            return demand_data
            
        except Exception as e:
            logger.error(f"Error fetching demand data for {commodity_name}: {str(e)}")
            return {
                'growth_rate': 0.02,
                'primary_consumers': [],
                'seasonal_factors': 'No data available',
                'recent_shifts': [],
                'uae_specific': 'No data available'
            }
    
    def get_geo_data(self, commodity_name, target_country='UAE'):
        """Get geopolitical and shipping data relevant to the commodity and target country"""
        cache_key = self._get_cache_key('get_geo_data', commodity_name, target_country)
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # In a production environment, this would use an actual API
            # For demo purposes, generating synthetic data
            
            # Primary exporters to UAE
            exporters_map = {
                'Oil': ['Saudi Arabia', 'Kuwait', 'Qatar'],
                'Gold': ['Switzerland', 'UK', 'USA', 'South Africa'],
                'Silver': ['UK', 'Switzerland', 'USA', 'Mexico'],
                'Copper': ['Chile', 'Peru', 'Australia', 'USA'],
                'Natural Gas': ['Qatar', 'Russia', 'USA', 'Australia'],
                'Wheat': ['Russia', 'Canada', 'USA', 'Australia', 'France'],
                'Coffee': ['Brazil', 'Vietnam', 'Colombia', 'Ethiopia']
            }
            
            # Trade agreements
            trade_agreements_map = {
                'Oil': ['GCC Economic Agreement', 'OPEC coordination'],
                'Gold': ['UAE-Switzerland trade agreements'],
                'Silver': ['Various bilateral trade agreements'],
                'Copper': ['Free trade agreements with producing countries'],
                'Natural Gas': ['LNG supply agreements with Qatar'],
                'Wheat': ['Agricultural trade agreements with major producers'],
                'Coffee': ['Trade agreements with coffee-producing nations']
            }
            
            # Shipping routes and risks
            shipping_routes_map = {
                'Oil': ['Strait of Hormuz', 'Persian Gulf'],
                'Gold': ['Air freight mainly', 'Secure shipping'],
                'Silver': ['Air freight and secure shipping'],
                'Copper': ['Sea routes via Suez Canal and around Africa'],
                'Natural Gas': ['LNG tankers, pipelines with neighboring states'],
                'Wheat': ['Bulk carriers via Suez Canal, Indian Ocean'],
                'Coffee': ['Container ships via multiple routes']
            }
            
            # Shipping risks
            shipping_risks_map = {
                'Oil': 'Moderate risk due to regional tensions',
                'Gold': 'Low risk, highly secure transport',
                'Silver': 'Low risk, secure transport',
                'Copper': 'Low to moderate risk from piracy and weather',
                'Natural Gas': 'Low risk for pipeline, moderate for LNG',
                'Wheat': 'Low risk, occasional weather delays',
                'Coffee': 'Low risk, standard shipping risks'
            }
            
            # Geopolitical factors
            geopolitical_map = {
                'Oil': ['Middle East tensions', 'OPEC+ production agreements'],
                'Gold': ['Global financial stability', 'Central bank policies'],
                'Silver': ['Industrial policy shifts', 'Economic uncertainty'],
                'Copper': ['Trade relations with Chile/Peru', 'Mining regulations'],
                'Natural Gas': ['Relations with Qatar', 'Global LNG market dynamics'],
                'Wheat': ['Food security policies', 'Trade restrictions by exporters'],
                'Coffee': ['Climate change affecting production', 'Producer country stability']
            }
            
            # Current risks score (0-1): 0 means low risk, 1 means high risk
            risk_score_map = {
                'Oil': 0.7,
                'Gold': 0.4,
                'Silver': 0.5,
                'Copper': 0.6,
                'Natural Gas': 0.5,
                'Wheat': 0.65,
                'Coffee': 0.5
            }
            
            geo_data = {
                'primary_exporters': exporters_map.get(commodity_name, []),
                'trade_agreements': trade_agreements_map.get(commodity_name, []),
                'shipping_routes': shipping_routes_map.get(commodity_name, []),
                'shipping_risks': shipping_risks_map.get(commodity_name, 'Standard shipping risks'),
                'geopolitical_factors': geopolitical_map.get(commodity_name, []),
                'risk_score': risk_score_map.get(commodity_name, 0.5)
            }
            
            self._set_in_cache(cache_key, geo_data)
            return geo_data
            
        except Exception as e:
            logger.error(f"Error fetching geopolitical data for {commodity_name}: {str(e)}")
            return {
                'primary_exporters': [],
                'trade_agreements': [],
                'shipping_routes': [],
                'shipping_risks': 'No data available',
                'geopolitical_factors': [],
                'risk_score': 0.5
            }
