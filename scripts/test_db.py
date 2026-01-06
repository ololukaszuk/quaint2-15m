#!/usr/bin/env python3
"""
Database connectivity test script
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services'))

from db_client import create_db_client_from_env
import constants


def main():
    print("=" * 60)
    print("Database Connectivity Test")
    print("=" * 60)
    
    try:
        # Create client
        print("\n1. Creating database client...")
        db = create_db_client_from_env()
        print("   ✓ Database client created")
        
        # Test connection
        print("\n2. Testing connection...")
        healthy = db.health_check()
        
        if healthy:
            print("   ✓ Database connection successful")
        else:
            print("   ✗ Database connection failed")
            return 1
        
        # Check tables
        print("\n3. Checking tables...")
        tables = [
            constants.TABLE_OHLCV_RAW,
            constants.TABLE_TECHNICAL_INDICATORS,
            constants.TABLE_PREDICTIONS,
            constants.TABLE_MODEL_REGISTRY
        ]
        
        for table in tables:
            exists = db.table_exists(table)
            status = "✓" if exists else "✗"
            print(f"   {status} {table}")
        
        # Get row counts
        print("\n4. Row counts:")
        for table in tables:
            try:
                result = db.fetch_one(f"SELECT COUNT(*) as count FROM {table}", dict_result=True)
                count = result['count'] if result else 0
                print(f"   {table}: {count} rows")
            except:
                print(f"   {table}: N/A")
        
        # Close
        db.close()
        
        print("\n" + "=" * 60)
        print("Database test complete!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
