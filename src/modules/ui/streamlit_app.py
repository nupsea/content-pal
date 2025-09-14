#!/usr/bin/env python3
"""
Streamlit UI for Content-Pal Movie Search and Feedback
"""
import streamlit as st
import sys
import os
import requests
import json
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..')
sys.path.append(src_path)

# Page configuration
st.set_page_config(
    page_title="Content-Pal",
    page_icon="🎬",
    layout="wide"
)

# App configuration - use config module for Docker/local detection
try:
    from .config import BASE_URL, RANDOM_QUERIES_FILE, DB_CONFIG
except ImportError:
    # Fallback for direct execution - detect if running in Docker
    import os
    IN_DOCKER = os.path.exists('/.dockerenv')
    
    if IN_DOCKER:
        # When running in Docker, connect to 'app' service
        BASE_URL = "http://app:5001"
    else:
        # When running locally, connect to localhost
        BASE_URL = "http://localhost:5001"
        
    RANDOM_QUERIES_FILE = "./data/ground_truth_retrieval.csv"
    DB_CONFIG = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'database': os.getenv('POSTGRES_DB', 'content_pal'), 
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'port': os.getenv('POSTGRES_PORT', '5432')
    }

def load_random_queries():
    """Load random queries from CSV file"""
    try:
        import pandas as pd
        if os.path.exists(RANDOM_QUERIES_FILE):
            df = pd.read_csv(RANDOM_QUERIES_FILE)
            return df['query'].tolist() if 'query' in df.columns else []
    except Exception as e:
        st.error(f"Could not load random queries: {e}")
    return []

def search_movies(query):
    """Call the movie recommendation API"""
    try:
        data = {"query": query}
        response = requests.post(f"{BASE_URL}/recommend", json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed with status {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(" Could not connect to the search service. Make sure it's running on port 5000.")
        return None
    except requests.exceptions.Timeout:
        st.error(" Search request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Search error: {e}")
        return None

def send_feedback(conversation_id, feedback_value):
    """Send feedback for a conversation"""
    try:
        data = {"conversation_id": conversation_id, "feedback": feedback_value}
        response = requests.post(f"{BASE_URL}/feedback", json=data, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Feedback error: {e}")
        return False

def get_database_stats():
    """Get statistics from PostgreSQL database"""
    try:
        # Add modules to path for database access
        import psycopg2
        from psycopg2.extras import DictCursor
        
        # Connect to database using config
        conn = psycopg2.connect(**DB_CONFIG)
        
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Total conversations
            cur.execute("SELECT COUNT(*) as total FROM conversations")
            total_conversations = cur.fetchone()['total']
            
            # Conversations today
            cur.execute("""
                SELECT COUNT(*) as today 
                FROM conversations 
                WHERE DATE(timestamp) = CURRENT_DATE
            """)
            conversations_today = cur.fetchone()['today']
            
            # Total feedback
            cur.execute("SELECT COUNT(*) as total FROM feedback")
            total_feedback = cur.fetchone()['total']
            
            # Feedback breakdown
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as negative
                FROM feedback
            """)
            feedback_breakdown = cur.fetchone()
            
            # Average response time
            cur.execute("SELECT AVG(response_time) as avg_time FROM conversations")
            avg_response_time = cur.fetchone()['avg_time'] or 0
            
            # Most used model
            cur.execute("""
                SELECT model_used, COUNT(*) as count 
                FROM conversations 
                GROUP BY model_used 
                ORDER BY count DESC 
                LIMIT 1
            """)
            most_used_model = cur.fetchone()
            
            # Relevance distribution
            cur.execute("""
                SELECT 
                    relevance,
                    COUNT(*) as count
                FROM conversations 
                GROUP BY relevance
                ORDER BY count DESC
            """)
            relevance_stats = cur.fetchall()
            
            # Recent queries
            cur.execute("""
                SELECT question, timestamp, relevance 
                FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_queries = cur.fetchall()
            
            conn.close()
            
            return {
                'total_conversations': total_conversations,
                'conversations_today': conversations_today,
                'total_feedback': total_feedback,
                'positive_feedback': feedback_breakdown['positive'] or 0,
                'negative_feedback': feedback_breakdown['negative'] or 0,
                'avg_response_time': float(avg_response_time),
                'most_used_model': most_used_model['model_used'] if most_used_model else 'Unknown',
                'relevance_stats': dict(relevance_stats),
                'recent_queries': recent_queries
            }
            
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def display_movie_recommendations(results_data):
    """Display movie recommendations in a nice format"""
    if not results_data:
        st.warning("No results returned from search")
        return None
    
    # Extract conversation ID for feedback
    conversation_id = results_data.get('conversation_id')
    
    # Parse the answer (it's a JSON string inside the response)
    answer = results_data.get('answer', '{}')
    if isinstance(answer, str):
        try:
            answer_data = json.loads(answer)
        except json.JSONDecodeError:
            st.error("Could not parse search results")
            return None
    else:
        answer_data = answer
    
    # Display metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Response Time", f"{results_data.get('response_time', 0):.2f}s")
    with col2:
        st.metric("Relevance", results_data.get('relevance', 'Unknown'))
    with col3:
        st.metric("Total Tokens", results_data.get('total_tokens', 0))
    
    # Display cost and explanation if available
    if 'openai_cost' in results_data:
        st.caption(f" Cost: ${results_data['openai_cost']:.4f}")
    if 'relevance_explanation' in results_data:
        with st.expander(" AI Relevance Assessment"):
            st.write(results_data['relevance_explanation'])
    
    # Display recommendations
    recommendations = answer_data.get('catalog_recommendations', [])
    
    if recommendations:
        st.subheader("Movie Recommendations")
        
        for i, recommendation in enumerate(recommendations, 1):
            with st.container():
                # Parse the recommendation to make movie title bold
                # Format is usually: "Title (Year): Description"
                # Simple pattern matching: look for "):" to find where title ends
                if '):' in recommendation:
                    # Split at "):" pattern to separate title from description
                    title_part, description_part = recommendation.split('):', 1)
                    title_part = title_part.strip() + ')'  # Add back the closing parenthesis
                    
                    st.markdown(f"#### {i}. {title_part}")
                    st.markdown(f"{description_part.strip()}", unsafe_allow_html=True)
                elif ':' in recommendation:
                    # Fallback for other formats
                    title_part, description_part = recommendation.split(':', 1)
                    st.markdown(f"#### {i}. {title_part.strip()}")
                    st.markdown(f"{description_part.strip()}", unsafe_allow_html=True)
                else:
                    # No colon found
                    st.markdown(f"#### {i}. {recommendation}")
                st.markdown("---")
    
    return conversation_id

def main():
    # Initialize session state
    if 'search_count' not in st.session_state:
        st.session_state.search_count = 0
    if 'feedback_count' not in st.session_state:
        st.session_state.feedback_count = 0
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'trigger_search' not in st.session_state:
        st.session_state.trigger_search = False
    
    # Header
    st.title("🎬 Content-Pal")
    st.markdown("*Intelligent search & recommendations for catalog movies and TV shows*")

    # Sidebar for settings and info
    with st.sidebar:
        st.header("⚙️ Service")
        
        # Connection status
        try:
            response = requests.get(f"{BASE_URL}/", timeout=2)
            if response.status_code == 200:
                st.success("✅ Search service connected")
            else:
                st.error("❌ Search service not responding properly")
        except Exception as e:
            st.error(f"❌ Could not connect to the search service at {BASE_URL}")
            st.caption("Start the service with: `docker-compose up`")
        
        st.markdown("---")
        
        # Database statistics
        st.subheader("📊 Database Stats")
        
        # Get database stats
        db_stats = get_database_stats()
        
        if db_stats:
            # Main metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Searches", db_stats['total_conversations'])
                st.metric("Today", db_stats['conversations_today'])
            with col2:
                st.metric("Total Feedback", db_stats['total_feedback'])
                st.metric("👍 Positive", db_stats['positive_feedback'])
            
            # Performance metrics
            st.metric("Avg Response", f"{db_stats['avg_response_time']:.2f}s")
            # st.metric("Main Model", db_stats['most_used_model'])
            
            # Relevance breakdown
            if db_stats['relevance_stats']:
                st.markdown("**Relevance Distribution:**")
                for relevance, count in db_stats['relevance_stats'].items():
                    percentage = (count / db_stats['total_conversations']) * 100 if db_stats['total_conversations'] > 0 else 0
                    st.caption(f"{relevance}: {count} ({percentage:.1f}%)")
            
            # Session stats (local to this session)
            st.markdown("---")
            st.subheader("🔄 This Session")
            st.metric("Session Searches", st.session_state.search_count)
            st.metric("Session Feedback", st.session_state.feedback_count)
            
        else:
            # Fallback to session-only stats if DB unavailable
            st.metric("Session Searches", st.session_state.search_count)
            st.metric("Session Feedback", st.session_state.feedback_count)
            st.caption("⚠️ Database stats unavailable")
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["🔍 Search", "📊 Analytics"])
    
    with tab2:
        st.header(" Database Analytics")
        
        # Refresh button
        if st.button("🔄 Refresh Stats"):
            st.rerun()
        
        # Get fresh stats
        db_stats = get_database_stats()
        
        if db_stats:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Conversations",
                    value=db_stats['total_conversations'],
                    delta=f"+{db_stats['conversations_today']} today"
                )
            
            with col2:
                feedback_rate = (db_stats['total_feedback'] / db_stats['total_conversations'] * 100) if db_stats['total_conversations'] > 0 else 0
                st.metric(
                    label="Feedback Rate", 
                    value=f"{feedback_rate:.1f}%",
                    delta=f"{db_stats['total_feedback']} total"
                )
            
            with col3:
                if db_stats['total_feedback'] > 0:
                    satisfaction = (db_stats['positive_feedback'] / db_stats['total_feedback'] * 100)
                    st.metric(
                        label="Satisfaction",
                        value=f"{satisfaction:.1f}%",
                        delta=f"{db_stats['positive_feedback']} positive"
                    )
                else:
                    st.metric("Satisfaction", "No data", "No feedback yet")
            
            with col4:
                st.metric(
                    label="Avg Response Time",
                    value=f"{db_stats['avg_response_time']:.2f}s",
                    delta=f"Model: {db_stats['most_used_model']}"
                )
            
            st.markdown("---")
            
            # Charts and detailed stats
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader(" Relevance Quality")
                if db_stats['relevance_stats']:
                    relevance_data = db_stats['relevance_stats']
                    
                    # Create a simple chart
                    import pandas as pd
                    chart_data = pd.DataFrame(
                        list(relevance_data.items()), 
                        columns=['Relevance', 'Count']
                    )
                    
                    st.bar_chart(chart_data.set_index('Relevance'))
                    
                    # Show percentages
                    total = sum(relevance_data.values())
                    for relevance, count in relevance_data.items():
                        percentage = (count / total * 100) if total > 0 else 0
                        st.caption(f"**{relevance}**: {count} searches ({percentage:.1f}%)")
                else:
                    st.info("No relevance data available yet")
            
            with col2:
                st.subheader(" Recent Activity")
                if db_stats['recent_queries']:
                    for i, query_data in enumerate(db_stats['recent_queries'], 1):
                        query = query_data['question']
                        timestamp = query_data['timestamp']
                        relevance = query_data['relevance']
                        
                        # Relevance emoji
                        relevance_emoji = {
                            'RELEVANT': '✅',
                            'PARTLY_RELEVANT': '⚠️', 
                            'NON_RELEVANT': '❌'
                        }.get(relevance, '❓')
                        
                        st.markdown(f"""
                        **{i}.** {query[:40]}{'...' if len(query) > 40 else ''}  
                        {relevance_emoji} {relevance} • {timestamp.strftime('%Y-%m-%d %H:%M')}
                        """)
                        st.markdown("---")
                else:
                    st.info("No recent queries found")
            
            # Cost and token analysis
            st.markdown("---")
            st.subheader("💰 Usage & Costs")
            
            # You could add more detailed cost analysis here
            total_cost_estimate = db_stats['total_conversations'] * 0.001  # Rough estimate
            st.metric("Estimated Total Cost", f"${total_cost_estimate:.3f}")
            st.caption(" Based on average token usage estimates")
            
        else:
            st.error("Could not connect to database for analytics")
            st.info("Make sure PostgreSQL is running and accessible")
    
    with tab1:
        st.header("Search for Movies & TV Shows")
        
        # Query input and random query button side by side
        random_queries = load_random_queries()
        if random_queries:
            col1, col2 = st.columns([3.5, 1])
            with col1:
                query = st.text_input(
                    "What would you like to watch?",
                    placeholder="e.g., Christopher Nolan movies, romantic comedies...",
                    value=st.session_state.current_query,
                    key="query_text_input"
                )
            with col2:
                # Add some vertical spacing to align button with input field
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("  🎲  Get Random Query ", key="random_query_btn", help="Get a random suggestion"):
                    import random
                    random_query = random.choice(random_queries)
                    st.session_state.current_query = random_query
                    st.session_state.trigger_search = True
                    st.rerun()
        else:
            query = st.text_input(
                "What would you like to watch?",
                placeholder="e.g., Christopher Nolan movies, romantic comedies...",
                value=st.session_state.current_query,
                key="query_text_input"
            )
        
        # Update session state when input changes
        if query != st.session_state.current_query:
            st.session_state.current_query = query
        
        # Example queries for inspiration
        with st.expander("💡 Example Queries"):
            examples = [
                "Feel good comedy movies",
                "mind-bending sci-fi movies",
                "dark comedy shows",
                "action movies from the 90s",
                "psychological thrillers",
                "Tom Hanks comedy movies"
            ]
            
            st.markdown("Click any example to try it:")
            
            # Create buttons in columns
            cols = st.columns(2)
            for i, example in enumerate(examples):
                col = cols[i % 2]
                if col.button(f" {example}", key=f"example_{i}"):
                    st.session_state.current_query = example
                    st.session_state.trigger_search = True
                    st.rerun()
        
        
        # Search button and results
        search_triggered = st.button("🔍 Search", type="primary") or st.session_state.trigger_search
        
        if search_triggered and st.session_state.current_query.strip():
            # Reset the trigger
            st.session_state.trigger_search = False
            
            with st.spinner("Searching for recommendations..."):
                start_time = time.time()
                results = search_movies(st.session_state.current_query.strip())
                search_time = time.time() - start_time
                
                if results:
                    st.session_state.search_count += 1
                    
                    # Display results
                    st.success(f" Search completed in {search_time:.2f} seconds")
                    conversation_id = display_movie_recommendations(results)
                    
                    # Feedback section
                    if conversation_id:
                        st.markdown("---")
                        st.subheader(" How was this recommendation?")
                        
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            if st.button("👍 Good", key=f"thumbs_up_{conversation_id}"):
                                if send_feedback(conversation_id, 1):
                                    st.success("Thanks for the positive feedback! 👍")
                                    st.session_state.feedback_count += 1
                                else:
                                    st.error("Failed to send feedback")
                        
                        with col2:
                            if st.button("👎 Not Good", key=f"thumbs_down_{conversation_id}"):
                                if send_feedback(conversation_id, -1):
                                    st.success("Thanks for the feedback! We'll improve. 👎")
                                    st.session_state.feedback_count += 1
                                else:
                                    st.error("Failed to send feedback")
                        
                        with col3:
                            st.caption("Your feedback helps improve our recommendations!")
        
        elif search_triggered and not st.session_state.current_query.strip():
            st.warning("Please enter a search query")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Content-Pal • RAG solution to seek movies & TV shows in the catalog •
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()