#!/usr/bin/env python3
import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

if not api_key or not api_secret:
    print("Error: BYBIT_API_KEY and BYBIT_API_SECRET must be set")
    exit(1)

exchange = ccxt.bybit({
    "apiKey": api_key,
    "secret": api_secret,
    "sandbox": False,
    "options": {"defaultType": "linear"},
})

try:
    # Fetch all positions (including closed ones)
    all_positions = exchange.fetch_positions()
    
    print(f"Total positions returned by API: {len(all_positions)}")
    print("\nPosition analysis:")
    
    open_count = 0
    for i, pos in enumerate(all_positions):
        contracts = pos.get("contracts")
        size = pos.get("size")
        symbol = pos.get("symbol")
        side = pos.get("side")
        
        # Check different ways a position might be "open"
        is_open_contracts = contracts and contracts > 0
        is_open_size = size and size > 0
        
        if is_open_contracts or is_open_size:
            open_count += 1
            print(f"{open_count}. {symbol} - Side: {side}, Contracts: {contracts}, Size: {size}")
            
    print(f"\nTotal open positions found: {open_count}")
    
except Exception as e:
    print(f"Error: {e}")