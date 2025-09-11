#!/usr/bin/env python3
"""
Debug the app.py workflow to see where the issue is
"""
import sys
sys.path.append('src')

from src.modules.workflow.rag import rag
from src.modules.workflow import db
import uuid

def debug_workflow():
    """Debug the full workflow from query to database save"""
    
    query = "Tom Cruise movies"
    print(f"1. Testing query: '{query}'")
    
    # Generate conversation ID
    conversation_id = str(uuid.uuid4())
    print(f"2. Generated conversation_id: {conversation_id}")
    
    # Test RAG function
    try:
        print("3. Calling rag() function...")
        recommendations = rag(query)
        print(f"4. RAG response keys: {list(recommendations.keys())}")
        print(f"5. RAG response preview: {str(recommendations)[:200]}...")
    except Exception as e:
        print(f"ERROR in rag(): {e}")
        return
    
    # Test database save
    try:
        print("6. Calling save_conversation()...")
        db.save_conversation(conversation_id, query, recommendations)
        print("7. save_conversation() completed successfully")
    except Exception as e:
        print(f"ERROR in save_conversation(): {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify data was saved
    try:
        print("8. Checking if data was saved...")
        import psycopg2
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            database=os.getenv('POSTGRES_DB', 'content_pal'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM conversations WHERE id = %s", (conversation_id,))
            count = cur.fetchone()[0]
            print(f"9. Found {count} records with conversation_id: {conversation_id}")
            
            cur.execute("SELECT COUNT(*) FROM conversations")
            total_count = cur.fetchone()[0]
            print(f"10. Total conversations in database: {total_count}")
            
        conn.close()
        
    except Exception as e:
        print(f"ERROR checking database: {e}")
        
    print("Debug completed!")

if __name__ == "__main__":
    debug_workflow()