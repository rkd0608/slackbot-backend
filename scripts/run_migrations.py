#!/usr/bin/env python3
"""Run database migrations on app startup."""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def run_migrations():
    """Run database migrations directly using SQL."""
    try:
        import psycopg2
        
        # Connect to database
        print("Connecting to database...")
        conn = psycopg2.connect(
            host="db",
            port=5432,
            user="postgres",
            password="admin",
            database="slackbot"
        )
        
        cursor = conn.cursor()
        
        # Check if tables already exist
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'workspace'
        """)
        
        if cursor.fetchone():
            print("Tables already exist, skipping migration")
            return
        
        print("Creating database tables...")
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("pgvector extension enabled")
        
        # Create workspace table
        cursor.execute("""
            CREATE TABLE workspace (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                name VARCHAR(255) NOT NULL,
                slack_id VARCHAR(50) NOT NULL UNIQUE,
                tokens JSONB NOT NULL DEFAULT '{}'
            )
        """)
        
        # Create user table
        cursor.execute("""
            CREATE TABLE "user" (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                workspace_id INTEGER NOT NULL REFERENCES workspace(id),
                slack_id VARCHAR(50) NOT NULL,
                name VARCHAR(255) NOT NULL,
                role VARCHAR(50) DEFAULT 'user'
            )
        """)
        
        # Create message table
        cursor.execute("""
            CREATE TABLE message (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                workspace_id INTEGER NOT NULL REFERENCES workspace(id),
                channel_id VARCHAR(50) NOT NULL,
                user_id INTEGER NOT NULL REFERENCES "user"(id),
                raw_payload JSONB NOT NULL
            )
        """)
        
        # Create knowledgeitem table
        cursor.execute("""
            CREATE TABLE knowledgeitem (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                workspace_id INTEGER NOT NULL REFERENCES workspace(id),
                title VARCHAR(500) NOT NULL,
                summary TEXT,
                content TEXT NOT NULL,
                confidence FLOAT NOT NULL DEFAULT 0.0,
                embedding TEXT
            )
        """)
        
        # Create query table
        cursor.execute("""
            CREATE TABLE query (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                workspace_id INTEGER NOT NULL REFERENCES workspace(id),
                user_id INTEGER NOT NULL REFERENCES "user"(id),
                text TEXT NOT NULL,
                response JSONB
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX ix_workspace_slack_id ON workspace(slack_id)")
        cursor.execute("CREATE INDEX ix_user_workspace_id ON \"user\"(workspace_id)")
        cursor.execute("CREATE INDEX ix_user_slack_id ON \"user\"(slack_id)")
        cursor.execute("CREATE INDEX ix_message_workspace_id ON message(workspace_id)")
        cursor.execute("CREATE INDEX ix_message_channel_id ON message(channel_id)")
        cursor.execute("CREATE INDEX ix_message_user_id ON message(user_id)")
        cursor.execute("CREATE INDEX ix_knowledgeitem_workspace_id ON knowledgeitem(workspace_id)")
        cursor.execute("CREATE INDEX ix_knowledgeitem_embedding ON knowledgeitem USING ivfflat (embedding vector_cosine_ops) WHERE embedding IS NOT NULL")
        cursor.execute("CREATE INDEX ix_query_workspace_id ON query(workspace_id)")
        cursor.execute("CREATE INDEX ix_query_user_id ON query(user_id)")
        
        # Commit changes
        conn.commit()
        print("Database tables created successfully!")
        
    except Exception as e:
        print(f"Error running migrations: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    asyncio.run(run_migrations())
