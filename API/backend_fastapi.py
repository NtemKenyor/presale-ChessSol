# ChessSol Presale Backend API v4
# FastAPI + SQLite implementation with enhanced CORS and header handling

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import List, Optional, Annotated
import sqlite3
import json
import os
from contextlib import contextmanager
import logging
from decimal import Decimal
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_FILE = "chesssol_presale.db"

# Constants
GOAL_AMOUNT_USD = 1000000  # $1M goal
SOL_TO_USD_RATE = 150  # Should be fetched from API in production
USDC_TO_USD_RATE = 1

# Database functions
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize the database with required tables"""
    with get_db_connection() as conn:
        # Main contributions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                tx_hash TEXT UNIQUE NOT NULL,
                method TEXT NOT NULL,
                network TEXT DEFAULT 'mainnet',
                timestamp TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_wallet ON contributions(wallet)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON contributions(timestamp)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_tx_hash ON contributions(tx_hash)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_network ON contributions(network)
        ''')
        
        conn.commit()

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    logger.info("ChessSol Presale API v4 started successfully")
    yield
    # Shutdown
    logger.info("ChessSol Presale API shutting down")

# FastAPI app initialization with lifespan
app = FastAPI(
    title="ChessSol Presale API v4",
    description="Enhanced Backend API for ChessSol token presale dashboard",
    version="4.0.0",
    lifespan=lifespan
)

# Enhanced CORS middleware to handle all frontend scenarios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Additional middleware to ensure CORS headers are always present
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Handle OPTIONS requests explicitly
@app.options("/{full_path:path}")
async def options_handler(request: Request):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Pydantic models for API requests/responses
class ContributionRequest(BaseModel):
    wallet: str
    amount: float
    currency: str
    tx_hash: str
    method: str  # 'crypto' or 'card'
    network: str = 'mainnet'  # 'mainnet' or 'dev'
    timestamp: str
    
    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v):
        if v not in ['SOL', 'USDC', 'USD']:
            raise ValueError('Currency must be SOL, USDC, or USD')
        return v
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        if v not in ['crypto', 'card']:
            raise ValueError('Method must be crypto or card')
        return v
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v, info):
        currency = info.data.get('currency')
        if currency == 'SOL' and v < 0.1:
            raise ValueError('Minimum SOL contribution is 0.1')
        elif currency == 'USDC' and v < 5.0:
            raise ValueError('Minimum USDC contribution is 5.0')
        elif currency == 'USD' and v < 10.0:
            raise ValueError('Minimum USD contribution is 10.0')
        return v

class ContributionResponse(BaseModel):
    id: int
    wallet: str
    amount: float
    currency: str
    tx_hash: str
    method: str
    network: str
    timestamp: str

class StatsResponse(BaseModel):
    total_sol: float
    total_usdc: float
    total_usd: float
    total_raised_usd: float
    contributor_count: int
    total_contributions: int
    progress_percentage: float

class ContributorStats(BaseModel):
    wallet: str
    amount: float
    currency: str
    method: str
    timestamp: str

def convert_to_usd(amount: float, currency: str) -> float:
    """Convert any currency to USD for calculations"""
    if currency == 'SOL':
        return amount * SOL_TO_USD_RATE
    elif currency == 'USDC':
        return amount * USDC_TO_USD_RATE
    elif currency == 'USD':
        return amount
    else:
        return 0

def get_network_from_header(x_network: Optional[str] = Header(None)) -> str:
    """Extract network from header, default to mainnet"""
    return x_network if x_network in ['dev', 'mainnet'] else 'mainnet'

# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ChessSol Presale API v4 is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "features": [
            "Enhanced CORS support",
            "Real-time contribution tracking",
            "Multi-network support",
            "Comprehensive error handling"
        ]
    }

@app.post("/contribute", response_model=dict)
async def record_contribution(
    contribution: ContributionRequest,
    network: str = Depends(get_network_from_header)
):
    """Record a new contribution"""
    try:
        # Override network if provided in request
        if contribution.network:
            network = contribution.network
            
        with get_db_connection() as conn:
            # Check if transaction hash already exists
            existing = conn.execute(
                "SELECT id FROM contributions WHERE tx_hash = ? AND network = ?",
                (contribution.tx_hash, network)
            ).fetchone()
            
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Transaction hash already exists for {network} network"
                )
            
            # Insert new contribution
            cursor = conn.execute('''
                INSERT INTO contributions (wallet, amount, currency, tx_hash, method, network, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                contribution.wallet,
                contribution.amount,
                contribution.currency,
                contribution.tx_hash,
                contribution.method,
                network,
                contribution.timestamp
            ))
            
            conn.commit()
            contribution_id = cursor.lastrowid
            
            logger.info(f"New contribution recorded: ID {contribution_id}, {contribution.amount} {contribution.currency} from {contribution.wallet[:10]}... on {network}")
            
            return {
                "success": True,
                "message": "Contribution recorded successfully",
                "contribution_id": contribution_id,
                "network": network,
                "amount": contribution.amount,
                "currency": contribution.currency,
                "usd_value": convert_to_usd(contribution.amount, contribution.currency)
            }
            
    except sqlite3.IntegrityError as e:
        logger.error(f"Database integrity error: {e}")
        raise HTTPException(status_code=400, detail="Transaction hash already exists")
    except Exception as e:
        logger.error(f"Error recording contribution: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats(network: str = Depends(get_network_from_header)):
    """Get current presale statistics"""
    try:
        with get_db_connection() as conn:
            # Get totals by currency
            stats = conn.execute('''
                SELECT 
                    currency,
                    SUM(amount) as total_amount,
                    COUNT(*) as contribution_count,
                    COUNT(DISTINCT wallet) as unique_contributors
                FROM contributions 
                WHERE network = ?
                GROUP BY currency
            ''', (network,)).fetchall()
            
            # Initialize totals
            total_sol = 0.0
            total_usdc = 0.0
            total_usd = 0.0
            total_contributions = 0
            
            # Process stats
            for stat in stats:
                if stat['currency'] == 'SOL':
                    total_sol = float(stat['total_amount'])
                elif stat['currency'] == 'USDC':
                    total_usdc = float(stat['total_amount'])
                elif stat['currency'] == 'USD':
                    total_usd = float(stat['total_amount'])
                total_contributions += stat['contribution_count']
            
            # Get unique contributor count
            contributor_count = conn.execute('''
                SELECT COUNT(DISTINCT wallet) as unique_count
                FROM contributions 
                WHERE network = ?
            ''', (network,)).fetchone()
            
            unique_contributors = contributor_count['unique_count'] if contributor_count else 0
            
            # Calculate total raised in USD
            total_raised_usd = (
                convert_to_usd(total_sol, 'SOL') +
                convert_to_usd(total_usdc, 'USDC') +
                convert_to_usd(total_usd, 'USD')
            )
            
            # Calculate progress percentage
            progress_percentage = min((total_raised_usd / GOAL_AMOUNT_USD) * 100, 100)
            
            logger.info(f"Stats requested for {network}: ${total_raised_usd:.2f} raised, {unique_contributors} contributors")
            
            return StatsResponse(
                total_sol=round(total_sol, 4),
                total_usdc=round(total_usdc, 2),
                total_usd=round(total_usd, 2),
                total_raised_usd=round(total_raised_usd, 2),
                contributor_count=unique_contributors,
                total_contributions=total_contributions,
                progress_percentage=round(progress_percentage, 2)
            )
            
    except Exception as e:
        logger.error(f"Error fetching stats for {network}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch statistics: {str(e)}")

@app.get("/contributions", response_model=List[ContributorStats])
async def get_contributions(
    limit: int = 50,
    offset: int = 0,
    network: str = Depends(get_network_from_header)
):
    """Get list of contributions for leaderboard"""
    try:
        with get_db_connection() as conn:
            contributions = conn.execute('''
                SELECT wallet, amount, currency, method, timestamp
                FROM contributions 
                WHERE network = ?
                ORDER BY 
                    CASE 
                        WHEN currency = 'SOL' THEN amount * ?
                        WHEN currency = 'USDC' THEN amount * ?
                        WHEN currency = 'USD' THEN amount
                        ELSE 0
                    END DESC,
                    timestamp DESC
                LIMIT ? OFFSET ?
            ''', (network, SOL_TO_USD_RATE, USDC_TO_USD_RATE, limit, offset)).fetchall()
            
            logger.info(f"Contributions requested for {network}: {len(contributions)} found")
            
            return [
                ContributorStats(
                    wallet=contrib['wallet'],
                    amount=float(contrib['amount']),
                    currency=contrib['currency'],
                    method=contrib['method'],
                    timestamp=contrib['timestamp']
                )
                for contrib in contributions
            ]
            
    except Exception as e:
        logger.error(f"Error fetching contributions for {network}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch contributions: {str(e)}")

@app.get("/contribution/{tx_hash}")
async def get_contribution_by_hash(
    tx_hash: str,
    network: str = Depends(get_network_from_header)
):
    """Get specific contribution by transaction hash"""
    try:
        with get_db_connection() as conn:
            contribution = conn.execute('''
                SELECT * FROM contributions 
                WHERE tx_hash = ? AND network = ?
            ''', (tx_hash, network)).fetchone()
            
            if not contribution:
                raise HTTPException(status_code=404, detail="Contribution not found")
            
            return {
                "id": contribution['id'],
                "wallet": contribution['wallet'],
                "amount": contribution['amount'],
                "currency": contribution['currency'],
                "tx_hash": contribution['tx_hash'],
                "method": contribution['method'],
                "network": contribution['network'],
                "timestamp": contribution['timestamp'],
                "created_at": contribution['created_at'],
                "usd_value": convert_to_usd(contribution['amount'], contribution['currency'])
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching contribution by hash: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch contribution: {str(e)}")

@app.get("/wallet/{wallet_address}")
async def get_wallet_contributions(
    wallet_address: str,
    network: str = Depends(get_network_from_header)
):
    """Get all contributions from a specific wallet"""
    try:
        with get_db_connection() as conn:
            contributions = conn.execute('''
                SELECT * FROM contributions 
                WHERE wallet = ? AND network = ?
                ORDER BY timestamp DESC
            ''', (wallet_address, network)).fetchall()
            
            # Calculate total contributed by this wallet in USD
            total_usd = 0
            contribution_list = []
            
            for contrib in contributions:
                usd_value = convert_to_usd(contrib['amount'], contrib['currency'])
                total_usd += usd_value
                
                contribution_list.append({
                    "id": contrib['id'],
                    "amount": contrib['amount'],
                    "currency": contrib['currency'],
                    "tx_hash": contrib['tx_hash'],
                    "method": contrib['method'],
                    "timestamp": contrib['timestamp'],
                    "usd_value": usd_value
                })
            
            return {
                "wallet": wallet_address,
                "network": network,
                "total_contributions": len(contributions),
                "total_usd_value": round(total_usd, 2),
                "contributions": contribution_list
            }
            
    except Exception as e:
        logger.error(f"Error fetching wallet contributions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch wallet contributions: {str(e)}")

@app.delete("/contribution/{contribution_id}")
async def delete_contribution(
    contribution_id: int,
    network: str = Depends(get_network_from_header)
):
    """Delete a contribution (admin only - add authentication in production)"""
    try:
        with get_db_connection() as conn:
            # Check if contribution exists
            existing = conn.execute('''
                SELECT * FROM contributions 
                WHERE id = ? AND network = ?
            ''', (contribution_id, network)).fetchone()
            
            if not existing:
                raise HTTPException(status_code=404, detail="Contribution not found")
            
            # Delete the contribution
            conn.execute('''
                DELETE FROM contributions 
                WHERE id = ? AND network = ?
            ''', (contribution_id, network))
            
            conn.commit()
            
            logger.info(f"Contribution {contribution_id} deleted from {network} network")
            
            return {
                "success": True,
                "message": f"Contribution {contribution_id} deleted successfully",
                "deleted_contribution": {
                    "id": existing['id'],
                    "wallet": existing['wallet'],
                    "amount": existing['amount'],
                    "currency": existing['currency']
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting contribution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete contribution: {str(e)}")

@app.get("/export")
async def export_contributions(
    format: str = "json",
    network: str = Depends(get_network_from_header)
):
    """Export all contributions (admin only - add authentication in production)"""
    try:
        with get_db_connection() as conn:
            contributions = conn.execute('''
                SELECT * FROM contributions 
                WHERE network = ?
                ORDER BY timestamp DESC
            ''', (network,)).fetchall()
            
            if format.lower() == "csv":
                # Return CSV format
                import io
                import csv
                from fastapi.responses import StreamingResponse
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(['ID', 'Wallet', 'Amount', 'Currency', 'TX Hash', 'Method', 'Network', 'Timestamp', 'Created At', 'USD Value'])
                
                # Write data
                for contrib in contributions:
                    usd_value = convert_to_usd(contrib['amount'], contrib['currency'])
                    writer.writerow([
                        contrib['id'],
                        contrib['wallet'],
                        contrib['amount'],
                        contrib['currency'],
                        contrib['tx_hash'],
                        contrib['method'],
                        contrib['network'],
                        contrib['timestamp'],
                        contrib['created_at'],
                        usd_value
                    ])
                
                output.seek(0)
                
                return StreamingResponse(
                    io.BytesIO(output.getvalue().encode()),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=chesssol_contributions_{network}.csv"}
                )
            else:
                # Return JSON format with enhanced data
                contributions_data = []
                total_usd = 0
                
                for contrib in contributions:
                    usd_value = convert_to_usd(contrib['amount'], contrib['currency'])
                    total_usd += usd_value
                    
                    contributions_data.append({
                        **dict(contrib),
                        'usd_value': usd_value
                    })
                
                return {
                    "network": network,
                    "export_date": datetime.now().isoformat(),
                    "total_contributions": len(contributions),
                    "total_usd_value": round(total_usd, 2),
                    "contributions": contributions_data
                }
            
    except Exception as e:
        logger.error(f"Error exporting contributions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export contributions: {str(e)}")

@app.get("/analytics/{network}")
async def get_analytics(network: str):
    """Get detailed analytics for a specific network"""
    try:
        with get_db_connection() as conn:
            # Daily contribution totals
            daily_stats = conn.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as contributions,
                    SUM(CASE WHEN currency = 'SOL' THEN amount * ? ELSE 
                        CASE WHEN currency = 'USDC' THEN amount * ? ELSE amount END END) as usd_total
                FROM contributions 
                WHERE network = ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 30
            ''', (SOL_TO_USD_RATE, USDC_TO_USD_RATE, network)).fetchall()
            
            # Method breakdown
            method_stats = conn.execute('''
                SELECT 
                    method,
                    COUNT(*) as count,
                    SUM(CASE WHEN currency = 'SOL' THEN amount * ? ELSE 
                        CASE WHEN currency = 'USDC' THEN amount * ? ELSE amount END END) as usd_total
                FROM contributions 
                WHERE network = ?
                GROUP BY method
            ''', (SOL_TO_USD_RATE, USDC_TO_USD_RATE, network)).fetchall()
            
            # Currency breakdown
            currency_stats = conn.execute('''
                SELECT 
                    currency,
                    COUNT(*) as count,
                    SUM(amount) as total_amount,
                    AVG(amount) as avg_amount
                FROM contributions 
                WHERE network = ?
                GROUP BY currency
            ''', (network,)).fetchall()
            
            return {
                "network": network,
                "generated_at": datetime.now().isoformat(),
                "daily_stats": [dict(row) for row in daily_stats],
                "method_breakdown": [dict(row) for row in method_stats],
                "currency_breakdown": [dict(row) for row in currency_stats]
            }
            
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")

# Enhanced error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found", 
            "status_code": 404,
            "path": str(request.url),
            "method": request.method
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "status_code": 500,
            "message": "An unexpected error occurred. Please try again later."
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.exception_handler(400)
async def bad_request_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad request",
            "status_code": 400,
            "detail": str(exc.detail) if hasattr(exc, 'detail') else "Invalid request"
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Development helper endpoint
@app.get("/dev/reset/{network}")
async def reset_network_data(network: str):
    """Reset all data for a specific network (development only)"""
    if network not in ['dev', 'testnet']:
        raise HTTPException(status_code=403, detail="Reset only allowed for dev/testnet networks")
    
    try:
        with get_db_connection() as conn:
            result = conn.execute('DELETE FROM contributions WHERE network = ?', (network,))
            deleted_count = result.rowcount
            conn.commit()
            
            logger.warning(f"RESET: Deleted {deleted_count} contributions from {network} network")
            
            return {
                "success": True,
                "message": f"Reset {network} network data",
                "deleted_contributions": deleted_count
            }
            
    except Exception as e:
        logger.error(f"Error resetting {network} data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # For direct script execution, disable reload to avoid the warning
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,  # Disable reload when running directly
        log_level="info"
    )