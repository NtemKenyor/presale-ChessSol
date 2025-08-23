# ChessSol Presale Backend API v4
# Flask + SQLite implementation with enhanced CORS and header handling

from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
from datetime import datetime
from typing import List, Optional
import sqlite3
import json
import os
from contextlib import contextmanager
import logging
from decimal import Decimal
import requests
import io
import csv
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Enhanced CORS configuration
CORS(app, origins="*", supports_credentials=True, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])


MAIN_RPC = os.getenv("mainnet")
DEVNET_RPC = os.getenv("devnet")
if not MAIN_RPC:
    raise ValueError("MAIN_RPC is not set in .env file!")
elif not DEVNET_RPC:
    raise ValueError("DEVNET_RPC is not set in .env file!")

# Database configuration
DATABASE_FILE = "chesssol_presale.db"

# Constants
GOAL_AMOUNT_USD = 1000000  # $1M goal
DEFAULT_SOL_TO_USD_RATE = 150  # Fallback rate
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

def get_solana_price():
    """Get live Solana price from API with fallback"""
    try:
        response = requests.get('https://chesssol.com/api/chesssol/backend/solana-price', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data.get('price', DEFAULT_SOL_TO_USD_RATE))
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning(f"Failed to fetch Solana price: {e}. Using fallback: {DEFAULT_SOL_TO_USD_RATE}")
    return DEFAULT_SOL_TO_USD_RATE

def convert_to_usd(amount: float, currency: str) -> float:
    """Convert any currency to USD for calculations"""
    if currency == 'SOL':
        sol_rate = get_solana_price()
        return amount * sol_rate
    elif currency == 'USDC':
        return amount * USDC_TO_USD_RATE
    elif currency == 'USD':
        return amount
    else:
        return 0

def get_network_from_header():
    """Extract network from header, default to mainnet"""
    x_network = request.headers.get('X-Network')
    return x_network if x_network in ['dev', 'mainnet'] else 'mainnet'

# Request validation functions
def validate_contribution_data(data):
    """Validate contribution data"""
    errors = []
    
    # Required fields
    required_fields = ['wallet', 'amount', 'currency', 'tx_hash', 'method', 'timestamp']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, errors
    
    # Currency validation
    if data['currency'] not in ['SOL', 'USDC', 'USD']:
        errors.append('Currency must be SOL, USDC, or USD')
    
    # Method validation
    if data['method'] not in ['crypto', 'card']:
        errors.append('Method must be crypto or card')
    
    # Amount validation
    currency = data['currency']
    amount = float(data['amount'])
    if currency == 'SOL' and amount < 0.1:
        errors.append('Minimum SOL contribution is 0.1')
    elif currency == 'USDC' and amount < 5.0:
        errors.append('Minimum USDC contribution is 5.0')
    elif currency == 'USD' and amount < 10.0:
        errors.append('Minimum USD contribution is 10.0')
    
    return len(errors) == 0, errors

# Initialize database on startup
with app.app_context():
    init_database()
    logger.info("ChessSol Presale API v4 started successfully")

# CORS preflight handler
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    return response

@app.route('/', methods=['GET', 'OPTIONS'])
def root():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    return jsonify({
        "message": "ChessSol Presale API v4 is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "features": [
            "Enhanced CORS support",
            "Real-time contribution tracking",
            "Multi-network support",
            "Comprehensive error handling",
            "Live Solana price integration"
        ]
    })

@app.route('/contribute', methods=['POST', 'OPTIONS'])
def record_contribution():
    """Record a new contribution"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate request data
        is_valid, errors = validate_contribution_data(data)
        if not is_valid:
            return jsonify({"error": "Validation failed", "details": errors}), 400
        
        network = get_network_from_header()
        # Override network if provided in request
        if 'network' in data and data['network']:
            network = data['network']
            
        with get_db_connection() as conn:
            # Check if transaction hash already exists
            existing = conn.execute(
                "SELECT id FROM contributions WHERE tx_hash = ? AND network = ?",
                (data['tx_hash'], network)
            ).fetchone()
            
            if existing:
                return jsonify({
                    "error": f"Transaction hash already exists for {network} network"
                }), 400
            
            # Insert new contribution
            cursor = conn.execute('''
                INSERT INTO contributions (wallet, amount, currency, tx_hash, method, network, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['wallet'],
                float(data['amount']),
                data['currency'],
                data['tx_hash'],
                data['method'],
                network,
                data['timestamp']
            ))
            
            conn.commit()
            contribution_id = cursor.lastrowid
            
            logger.info(f"New contribution recorded: ID {contribution_id}, {data['amount']} {data['currency']} from {data['wallet'][:10]}... on {network}")
            
            return jsonify({
                "success": True,
                "message": "Contribution recorded successfully",
                "contribution_id": contribution_id,
                "network": network,
                "amount": float(data['amount']),
                "currency": data['currency'],
                "usd_value": convert_to_usd(float(data['amount']), data['currency'])
            })
            
    except sqlite3.IntegrityError as e:
        logger.error(f"Database integrity error: {e}")
        return jsonify({"error": "Transaction hash already exists"}), 400
    except Exception as e:
        logger.error(f"Error recording contribution: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/stats', methods=['GET', 'OPTIONS'])
def get_stats():
    """Get current presale statistics"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        network = get_network_from_header()
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
            
            return jsonify({
                "total_sol": round(total_sol, 4),
                "total_usdc": round(total_usdc, 2),
                "total_usd": round(total_usd, 2),
                "total_raised_usd": round(total_raised_usd, 2),
                "contributor_count": unique_contributors,
                "total_contributions": total_contributions,
                "progress_percentage": round(progress_percentage, 2)
            })
            
    except Exception as e:
        logger.error(f"Error fetching stats for {network}: {e}")
        return jsonify({"error": f"Failed to fetch statistics: {str(e)}"}), 500

@app.route('/contributions', methods=['GET', 'OPTIONS'])
def get_contributions():
    """Get list of contributions for leaderboard"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        network = get_network_from_header()
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        sol_rate = get_solana_price()
        
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
            ''', (network, sol_rate, USDC_TO_USD_RATE, limit, offset)).fetchall()
            
            logger.info(f"Contributions requested for {network}: {len(contributions)} found")
            
            result = []
            for contrib in contributions:
                result.append({
                    "wallet": contrib['wallet'],
                    "amount": float(contrib['amount']),
                    "currency": contrib['currency'],
                    "method": contrib['method'],
                    "timestamp": contrib['timestamp']
                })
            
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error fetching contributions for {network}: {e}")
        return jsonify({"error": f"Failed to fetch contributions: {str(e)}"}), 500

@app.route("/rpc_main", methods=["POST"])
def proxy_rpc_main():
    try:
        # Forward JSON-RPC request to QuickNode
        response = requests.post(
            MAIN_RPC,
            json=request.json,
            headers={"Content-Type": "application/json"}
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rpc_dev", methods=["POST"])
def proxy_rpc_dev():
    try:
        # Forward JSON-RPC request to QuickNode
        response = requests.post(
            DEVNET_RPC,
            json=request.json,
            headers={"Content-Type": "application/json"}
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/contribution/<tx_hash>', methods=['GET', 'OPTIONS'])
def get_contribution_by_hash(tx_hash):
    """Get specific contribution by transaction hash"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        network = get_network_from_header()
        with get_db_connection() as conn:
            contribution = conn.execute('''
                SELECT * FROM contributions 
                WHERE tx_hash = ? AND network = ?
            ''', (tx_hash, network)).fetchone()
            
            if not contribution:
                return jsonify({"error": "Contribution not found"}), 404
            
            return jsonify({
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
            })
            
    except Exception as e:
        logger.error(f"Error fetching contribution by hash: {e}")
        return jsonify({"error": f"Failed to fetch contribution: {str(e)}"}), 500

@app.route('/wallet/<wallet_address>', methods=['GET', 'OPTIONS'])
def get_wallet_contributions(wallet_address):
    """Get all contributions from a specific wallet"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        network = get_network_from_header()
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
            
            return jsonify({
                "wallet": wallet_address,
                "network": network,
                "total_contributions": len(contributions),
                "total_usd_value": round(total_usd, 2),
                "contributions": contribution_list
            })
            
    except Exception as e:
        logger.error(f"Error fetching wallet contributions: {e}")
        return jsonify({"error": f"Failed to fetch wallet contributions: {str(e)}"}), 500

@app.route('/contribution/<int:contribution_id>', methods=['DELETE', 'OPTIONS'])
def delete_contribution(contribution_id):
    """Delete a contribution (admin only - add authentication in production)"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        network = get_network_from_header()
        with get_db_connection() as conn:
            # Check if contribution exists
            existing = conn.execute('''
                SELECT * FROM contributions 
                WHERE id = ? AND network = ?
            ''', (contribution_id, network)).fetchone()
            
            if not existing:
                return jsonify({"error": "Contribution not found"}), 404
            
            # Delete the contribution
            conn.execute('''
                DELETE FROM contributions 
                WHERE id = ? AND network = ?
            ''', (contribution_id, network))
            
            conn.commit()
            
            logger.info(f"Contribution {contribution_id} deleted from {network} network")
            
            return jsonify({
                "success": True,
                "message": f"Contribution {contribution_id} deleted successfully",
                "deleted_contribution": {
                    "id": existing['id'],
                    "wallet": existing['wallet'],
                    "amount": existing['amount'],
                    "currency": existing['currency']
                }
            })
            
    except Exception as e:
        logger.error(f"Error deleting contribution: {e}")
        return jsonify({"error": f"Failed to delete contribution: {str(e)}"}), 500

@app.route('/export', methods=['GET', 'OPTIONS'])
def export_contributions():
    """Export all contributions (admin only - add authentication in production)"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        network = get_network_from_header()
        format = request.args.get('format', 'json')
        
        with get_db_connection() as conn:
            contributions = conn.execute('''
                SELECT * FROM contributions 
                WHERE network = ?
                ORDER BY timestamp DESC
            ''', (network,)).fetchall()
            
            if format.lower() == "csv":
                # Return CSV format
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
                
                # Create response with CSV data
                response = make_response(output.getvalue())
                response.headers['Content-Type'] = 'text/csv'
                response.headers['Content-Disposition'] = f'attachment; filename=chesssol_contributions_{network}.csv'
                return response
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
                
                return jsonify({
                    "network": network,
                    "export_date": datetime.now().isoformat(),
                    "total_contributions": len(contributions),
                    "total_usd_value": round(total_usd, 2),
                    "contributions": contributions_data
                })
            
    except Exception as e:
        logger.error(f"Error exporting contributions: {e}")
        return jsonify({"error": f"Failed to export contributions: {str(e)}"}), 500

@app.route('/analytics/<network>', methods=['GET', 'OPTIONS'])
def get_analytics(network):
    """Get detailed analytics for a specific network"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        sol_rate = get_solana_price()
        
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
            ''', (sol_rate, USDC_TO_USD_RATE, network)).fetchall()
            
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
            ''', (sol_rate, USDC_TO_USD_RATE, network)).fetchall()
            
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
            
            return jsonify({
                "network": network,
                "generated_at": datetime.now().isoformat(),
                "daily_stats": [dict(row) for row in daily_stats],
                "method_breakdown": [dict(row) for row in method_stats],
                "currency_breakdown": [dict(row) for row in currency_stats]
            })
            
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return jsonify({"error": f"Failed to fetch analytics: {str(e)}"}), 500

@app.route('/dev/reset/<network>', methods=['GET', 'OPTIONS'])
def reset_network_data(network):
    """Reset all data for a specific network (development only)"""
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if network not in ['dev', 'testnet']:
        return jsonify({"error": "Reset only allowed for dev/testnet networks"}), 403
    
    try:
        with get_db_connection() as conn:
            result = conn.execute('DELETE FROM contributions WHERE network = ?', (network,))
            deleted_count = result.rowcount
            conn.commit()
            
            logger.warning(f"RESET: Deleted {deleted_count} contributions from {network} network")
            
            return jsonify({
                "success": True,
                "message": f"Reset {network} network data",
                "deleted_contributions": deleted_count
            })
            
    except Exception as e:
        logger.error(f"Error resetting {network} data: {e}")
        return jsonify({"error": f"Failed to reset data: {str(e)}"}), 500

# Error handlers
@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({
        "error": "Endpoint not found", 
        "status_code": 404,
        "path": request.path,
        "method": request.method
    }), 404

@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"Internal server error on {request.method} {request.path}: {e}")
    return jsonify({
        "error": "Internal server error", 
        "status_code": 500,
        "message": "An unexpected error occurred. Please try again later."
    }), 500

@app.errorhandler(400)
def bad_request_handler(e):
    return jsonify({
        "error": "Bad request",
        "status_code": 400,
        "detail": str(e) if hasattr(e, 'description') else "Invalid request"
    }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)