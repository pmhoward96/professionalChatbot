import streamlit as st
import mlflow
import mlflow.deployments
from typing import List, Dict, Any
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
from datetime import datetime
import os
import traceback
from src.data.professional_data import load_professional_data

# Configuration for Databricks Native Setup
class Config:
    # Databricks foundation model endpoints (adjust based on what's available)
    FOUNDATION_MODEL_ENDPOINT = "databricks-llama-4-maverick"  # or available model
    EMBEDDING_MODEL = "databricks-bge-large-en"  # Use Databricks embedding model if available
    FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fallback to sentence-transformers
    MAX_CONTEXT_LENGTH = 3000

    try:
        DATABRICKS_HOST = dbutils.secrets.get("portfolio", "databricks_host")
        DATABRICKS_TOKEN = dbutils.secrets.get("portfolio", "databricks_token")
        SOURCE = "databricks_secrets"
    except Exception:
        DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
        DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
        SOURCE = "env"

# Initialize Databricks clients
@st.cache_resource
def get_databricks_client():
    """Initialize Databricks MLflow deployment client"""
    st.sidebar.info(f"Databricks Host: {Config.DATABRICKS_HOST}")
    st.sidebar.info(f"Databricks Token: {'SET' if Config.DATABRICKS_TOKEN else 'NOT SET'}")
    st.sidebar.info(f"Config Source: {Config.SOURCE}")
    try:
        return mlflow.deployments.get_deploy_client("databricks")
    except Exception as e:
        st.error(f"Failed to initialize Databricks client: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    """Load embedding model - try Databricks first, fallback to sentence-transformers"""
    try:
        client = get_databricks_client()
        if client:
            st.sidebar.info(f"Embedding endpoint: {Config.EMBEDDING_MODEL}")
            # Test if embeddings endpoint is available
            test_response = client.predict(
                endpoint=Config.EMBEDDING_MODEL,  # Common Databricks embedding model
                inputs={"input": ["test"]}
            )
            st.success("✅ Databricks embedding model connected successfully")
            return "databricks"
    except Exception as e:
        st.warning(f"⚠️ Databricks embeddings not available: {e}")
        st.info("🔄 Falling back to sentence-transformers...")
    
    # Fallback to sentence-transformers
    try:
        model = SentenceTransformer(Config.FALLBACK_EMBEDDING_MODEL)
        st.success(f"✅ Loaded fallback embedding model: {Config.FALLBACK_EMBEDDING_MODEL}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        return None

@st.cache_resource
def get_professional_data():
    """Load professional experience data"""
    return load_professional_data()

class DatabricksRAG:
    """RAG implementation using Databricks native capabilities"""
    
    def __init__(self):
        self.client = get_databricks_client()
        embedding_result = load_embedding_model()
        if embedding_result == "databricks":
            self.embedding_model = "databricks"
            self.fallback_model = SentenceTransformer(Config.FALLBACK_EMBEDDING_MODEL)
        else:
            self.embedding_model = "local"
            self.fallback_model = embedding_result
        self.professional_data = get_professional_data()
        self.documents = []
        self.embeddings = []
        self.setup_knowledge_base()
    
    def setup_knowledge_base(self):
        """Create knowledge base from professional data"""
        self.documents = self._create_documents()
        self.embeddings = self._create_embeddings()
    
    def _create_documents(self) -> List[str]:
        """Convert professional data into searchable document chunks"""
        documents = []
        
        # Add experience entries
        for exp in self.professional_data["experience"]:
            doc = f"Professional Experience:\n"
            doc += f"Role: {exp['role']} at {exp['company']} ({exp['duration']})\n"
            doc += f"Description: {exp['description']}\n"
            doc += f"Key Skills: {', '.join(exp['skills'])}"
            documents.append(doc)
        
        # Add project entries
        for proj in self.professional_data["projects"]:
            doc = f"Project Experience:\n"
            doc += f"Project: {proj['name']}\n"
            doc += f"Description: {proj['description']}\n"
            doc += f"Technology Stack: {', '.join(proj['tech_stack'])}\n"
            doc += f"Business Impact: {proj['impact']}"
            documents.append(doc)
        
        # Add skills by category
        for category, skills in self.professional_data["skills"].items():
            doc = f"Technical Skills - {category.replace('_', ' ').title()}:\n"
            doc += f"Skills: {', '.join(skills)}\n"
            doc += f"Proficiency: Expert level in {category.replace('_', ' ')}"
            documents.append(doc)
        
        # Add certifications
        if "certifications" in self.professional_data:
            for cert in self.professional_data["certifications"]:
                doc = f"Professional Certification:\n"
                doc += f"Certification: {cert['name']} ({cert['year']})\n"
                doc += f"Description: {cert['description']}"
                documents.append(doc)
        
        # Add education
        edu = self.professional_data["education"]
        edu_doc = f"Educational Background:\n"
        edu_doc += f"Degree: {edu['degree']} from {edu['university']} ({edu['year']})\n"
        edu_doc += f"Relevant Coursework: {', '.join(edu['relevant_coursework'])}"
        documents.append(edu_doc)
        
        return documents
    
    def _create_embeddings(self) -> List[np.ndarray]:
        """Create embeddings using Databricks or fallback model"""
        embeddings = []
        
        try:
            for i, document in enumerate(self.documents):
                try:
                    if self.embedding_model == "databricks" and self.client:
                        response = self.client.predict(
                            endpoint=Config.EMBEDDING_MODEL,  # Use the config value
                            inputs={"input": document}
                        )
                        
                        # Handle different response formats
                        if isinstance(response, dict):
                            if "data" in response and isinstance(response["data"], list):
                                if "embedding" in response["data"][0]:
                                    embedding = np.array(response["data"][0]["embedding"])
                                else:
                                    embedding = np.array(response["data"][0])
                            elif "embedding" in response:
                                embedding = np.array(response["embedding"])
                            else:
                                # Fallback to local model
                                embedding = self.fallback_model.encode(document)
                        else:
                            embedding = np.array(response)
                    else:
                        # Use the fallback model
                        embedding = self.fallback_model.encode(document)
                    
                    embeddings.append(embedding)
                except Exception as e:
                    st.sidebar.warning(f"Error embedding document {i}: {str(e)}")
                    embeddings.append(self.fallback_model.encode(document))
                    
            return embeddings
        except Exception as e:
            st.sidebar.error(f"Error in _create_embeddings: {str(e)}")
            st.sidebar.error(traceback.format_exc())
            return [self.fallback_model.encode(doc) for doc in self.documents]
    
    def retrieve_context(self, query: str, top_k: int = 3, max_context_tokens: int = 1500) -> str:
        """Retrieve relevant context for a query using similarity search with token management"""
        try:
            # Create query embedding
            if self.embedding_model == "databricks" and self.client:
                try:
                    query_response = self.client.predict(
                        endpoint=Config.EMBEDDING_MODEL,  # Use config value
                        inputs={"input": query}
                    )
                    # Handle different possible response formats
                    if isinstance(query_response, dict):
                        if "data" in query_response and isinstance(query_response["data"], list):
                            if "embedding" in query_response["data"][0]:
                                query_embedding = np.array(query_response["data"][0]["embedding"])
                            else:
                                query_embedding = np.array(query_response["data"][0])
                        elif "embedding" in query_response:
                            query_embedding = np.array(query_response["embedding"])
                        else:
                            # Fallback to local model if format is unexpected
                            query_embedding = self.fallback_model.encode(query)
                    else:
                        query_embedding = np.array(query_response)
                except Exception as e:
                    st.sidebar.error(f"Error with Databricks embedding: {str(e)}")
                    query_embedding = self.fallback_model.encode(query)
            else:
                query_embedding = self.fallback_model.encode(query)
        
            # Ensure query_embedding is a numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
                
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                # Ensure doc_embedding is also a numpy array
                if not isinstance(doc_embedding, np.ndarray):
                    doc_embedding = np.array(doc_embedding)
                    
                # Compute cosine similarity
                norm_q = np.linalg.norm(query_embedding)
                norm_d = np.linalg.norm(doc_embedding)
                
                if norm_q > 0 and norm_d > 0:  # Avoid division by zero
                    similarity = np.dot(query_embedding, doc_embedding) / (norm_q * norm_d)
                else:
                    similarity = 0
                    
                similarities.append((i, similarity))
        
            # Sort by similarity and retrieve top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in similarities[:top_k]]
            
            # Build context from selected chunks - FIXED
            context_parts = []
            total_tokens = 0
            
            for idx in selected_indices:
                # Use documents directly
                chunk_text = self.documents[idx]
                tokens = len(chunk_text.split())  # Simple token estimation
                
                if total_tokens + tokens <= max_context_tokens:
                    context_parts.append(chunk_text)
                    total_tokens += tokens
                else:
                    break
                
            return "\n\n".join(context_parts)
        except Exception as e:
            st.sidebar.error(f"Error in retrieve_context: {str(e)}")
            st.sidebar.error(traceback.format_exc())
            return "Error retrieving context."
    
    def get_contextual_summary(self, query_type: str) -> str:
        """Get a focused summary based on query type"""
        if "project" in query_type.lower():
            return self._get_projects_summary()
        elif "skill" in query_type.lower() or "technology" in query_type.lower():
            return self._get_skills_summary()
        elif "experience" in query_type.lower() or "role" in query_type.lower():
            return self._get_experience_summary()
        else:
            return self._get_general_summary()
    
    def _get_projects_summary(self) -> str:
        """Focused project summary"""
        projects = self.professional_data["projects"]
        summary = "Key Projects:\n"
        for proj in projects[:2]:  # Limit to top 2 projects
            summary += f"• {proj['name']}: {proj['description'][:150]}...\n"
        return summary
    
    def _get_skills_summary(self) -> str:
        """Focused skills summary"""
        skills = self.professional_data["skills"]
        summary = "Technical Skills:\n"
        for category, skill_list in skills.items():
            if category in ["databricks_platform", "programming", "ml_frameworks"]:
                summary += f"• {category.replace('_', ' ').title()}: {', '.join(skill_list[:5])}\n"
        return summary
    
    def _get_experience_summary(self) -> str:
        """Focused experience summary"""
        experiences = self.professional_data["experience"]
        summary = "Professional Experience:\n"
        for exp in experiences[:2]:  # Limit to recent experiences
            summary += f"• {exp['role']} at {exp['company']}: {exp['description'][:150]}...\n"
        return summary
    
    def _get_general_summary(self) -> str:
        """General professional summary"""
        return """Professional Overview:
• Senior Data Scientist/Engineer with extensive Databricks expertise
• 4+ years in ML/Data Engineering with focus on Lakehouse architecture
• Certified in Databricks Data Engineering and ML Professional
• Experience with end-to-end ML pipelines, real-time analytics, and MLOps"""

class DatabricksChatBot:
    """Chatbot using Databricks foundation models"""
    
    def __init__(self):
        self.client = get_databricks_client()
        self.rag = DatabricksRAG()
        
    def generate_response(self, user_query: str, chat_history: List[Dict]) -> str:
        """Generate response using Databricks foundation model with smart context management"""
        
        if not self.client:
            return "I'm sorry, but I'm having trouble connecting to Databricks services right now."
        
        # Analyze query to determine focus area
        query_lower = user_query.lower()
        
        # Get focused context based on query type
        if any(word in query_lower for word in ["project", "built", "developed", "created"]):
            context = self.rag.retrieve_context(user_query, top_k=2, max_context_tokens=1000)
            context += "\n\n" + self.rag.get_contextual_summary("project")
        elif any(word in query_lower for word in ["skill", "technology", "tool", "programming"]):
            context = self.rag.retrieve_context(user_query, top_k=2, max_context_tokens=800)
            context += "\n\n" + self.rag.get_contextual_summary("skills")
        elif any(word in query_lower for word in ["experience", "role", "job", "work"]):
            context = self.rag.retrieve_context(user_query, top_k=2, max_context_tokens=1000)
            context += "\n\n" + self.rag.get_contextual_summary("experience")
        else:
            # General query - use standard retrieval
            context = self.rag.retrieve_context(user_query, top_k=3, max_context_tokens=1200)
        
        # Create focused system message
        system_message = f"""You are a professional AI assistant representing a skilled data scientist and ML engineer with extensive Databricks experience.

Based on the following context, answer questions about their professional background in a conversational and enthusiastic manner.

Context:
{context}

Guidelines:
- Be specific and reference concrete examples from the context
- Show enthusiasm for their Databricks expertise
- If the context doesn't contain relevant information, provide a general response about their capabilities
- Keep responses focused and under 200 words
- Highlight unique value propositions
"""
        
        # Prepare messages - keep chat history minimal for performance
        messages = [{"role": "system", "content": system_message}]
        
        # Only include last 2 exchanges to preserve context window
        recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
        messages.extend(recent_history)
        
        # Add current user query
        messages.append({"role": "user", "content": user_query})
        
        try:
            # Call Databricks foundation model
            response = self.client.predict(
                endpoint=Config.FOUNDATION_MODEL_ENDPOINT,
                inputs={
                    "messages": messages,
                    "max_tokens": 300,  # Reduced for better performance
                    "temperature": 0.7
                }
            )
            
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request. Please try again. Error: {str(e)}"

# Streamlit App
def main():
    st.set_page_config(
        page_title="Databricks Professional AI Assistant",
        page_icon="🔥",
        layout="wide"
    )
    
    # Header with Databricks branding
    st.title("🔥 Databricks Professional AI Assistant")
    st.markdown("*Powered by Databricks Foundation Models and MLflow*")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading Databricks-powered knowledge base..."):
            st.session_state.chatbot = DatabricksChatBot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm here to discuss my professional background in data science and machine learning, with a focus on Databricks platform expertise. Feel free to ask about my experience with Databricks, MLflow, Delta Lake, or any other aspects of my professional journey!"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about my Databricks expertise and professional experience..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking with Databricks AI..."):
                response = st.session_state.chatbot.generate_response(
                    prompt, 
                    st.session_state.messages[:-1]
                )
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with Databricks-focused information
    with st.sidebar:
        st.header("🔥 Databricks Demo")
        st.markdown("""
        This chatbot showcases:
        - **Databricks Foundation Models** for natural language generation
        - **MLflow** for model management and deployment
        - **Native Databricks Integration** within the platform
        - **RAG with Databricks** for context-aware responses
        - **Lakehouse Architecture** knowledge demonstration
        
        **Databricks Technologies Demonstrated:**
        - Foundation Model APIs
        - MLflow Deployments
        - Delta Lake (in professional context)
        - Unity Catalog (in experience)
        - Delta Live Tables (in projects)
        - Structured Streaming (in projects)
        """)
        
        st.header("Professional Focus")
        st.markdown("""
        **Databricks Expertise:**
        - Data Engineering with Delta Lake
        - MLOps with MLflow
        - Real-time Analytics
        - Lakehouse Architecture
        - Unity Catalog Governance
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm here to discuss my professional background in data science and machine learning, with a focus on Databricks platform expertise. Feel free to ask about my experience with Databricks, MLflow, Delta Lake, or any other aspects of my professional journey!"}
            ]
            st.rerun()

if __name__ == "__main__":
    main()