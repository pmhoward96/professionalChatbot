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

# Configuration for Databricks Native Setup
class Config:
    # Databricks foundation model endpoints (adjust based on what's available)
    FOUNDATION_MODEL_ENDPOINT = "databricks-gemma-3-12b"  # or available model
    EMBEDDING_MODEL = "databricks-bge-large-en"  # Use Databricks embedding model if available
    FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fallback to sentence-transformers
    MAX_CONTEXT_LENGTH = 3000
    DATABRICKS_HOST = None  # Will be auto-detected
    DATABRICKS_TOKEN = None  # Will use default authentication

# Initialize Databricks clients
@st.cache_resource
def get_databricks_client():
    """Initialize Databricks MLflow deployment client"""
    try:
        return mlflow.deployments.get_deploy_client("databricks")
    except Exception as e:
        st.error(f"Failed to initialize Databricks client: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    """Load embedding model - try Databricks first, fallback to sentence-transformers"""
    try:
        # Try to use Databricks embedding endpoint first
        client = get_databricks_client()
        if client:
            # Test if embeddings endpoint is available
            test_response = client.predict(
                endpoint="databricks-bge-large-en",  # Common Databricks embedding model
                inputs={"input": ["test"]}
            )
            return "databricks"
    except:
        pass
    
    # Fallback to sentence-transformers
    return SentenceTransformer(Config.FALLBACK_EMBEDDING_MODEL)

@st.cache_resource
def load_professional_data():
    """Load professional experience data"""
    professional_data = {
        "experience": [
            {
                "role": "Senior Data Scientist",
                "company": "TechCorp Inc.",
                "duration": "2022-2024",
                "description": "Led machine learning initiatives using Databricks, developed predictive models with MLflow, managed team of 3 junior data scientists. Implemented end-to-end ML pipelines with Delta Lake and Unity Catalog.",
                "skills": ["Python", "SQL", "Databricks", "MLflow", "Delta Lake", "Unity Catalog", "Team Leadership"]
            },
            {
                "role": "Data Engineer", 
                "company": "DataFlow Solutions",
                "duration": "2020-2022",
                "description": "Built data pipelines on Databricks platform, created real-time streaming analytics with Delta Live Tables, performed large-scale data processing with Apache Spark.",
                "skills": ["Python", "Apache Spark", "Delta Live Tables", "SQL", "ETL", "Databricks"]
            }
        ],
        "projects": [
            {
                "name": "Customer Analytics Platform on Databricks",
                "description": "Built comprehensive customer analytics platform using Databricks Lakehouse architecture. Implemented real-time feature engineering with Delta Live Tables and MLOps workflows with MLflow.",
                "tech_stack": ["Databricks", "Delta Lake", "MLflow", "Apache Spark", "Python", "Unity Catalog"],
                "impact": "Reduced time-to-insights by 70% and improved model deployment efficiency by 50%"
            },
            {
                "name": "Real-time Fraud Detection with Databricks",
                "description": "Developed streaming fraud detection system using Databricks Structured Streaming and MLflow Model Registry. Processes 100k+ transactions per minute with sub-second latency.",
                "tech_stack": ["Databricks", "Structured Streaming", "MLflow", "Delta Lake", "Python", "Kafka"],
                "impact": "Reduced fraud losses by 45% and improved detection accuracy to 99.2%"
            }
        ],
        "skills": {
            "databricks_platform": ["Databricks Workspace", "Delta Lake", "MLflow", "Unity Catalog", "Delta Live Tables"],
            "programming": ["Python", "SQL", "Scala", "R"],
            "ml_frameworks": ["MLflow", "Scikit-learn", "TensorFlow", "PyTorch", "XGBoost"],
            "data_engineering": ["Apache Spark", "Structured Streaming", "Delta Live Tables", "Apache Kafka"],
            "cloud_platforms": ["Databricks", "AWS", "Azure"],
            "mlops": ["MLflow", "Model Registry", "Automated Retraining", "A/B Testing"]
        },
        "certifications": [
            {
                "name": "Databricks Certified Data Engineer Professional",
                "year": "2023",
                "description": "Advanced certification in Databricks platform and data engineering"
            },
            {
                "name": "Databricks Certified Machine Learning Professional", 
                "year": "2023",
                "description": "Expert-level MLOps and machine learning on Databricks platform"
            }
        ],
        "education": {
            "degree": "Master of Science in Data Science",
            "university": "University XYZ",
            "year": "2020",
            "relevant_coursework": ["Machine Learning", "Big Data Systems", "Statistical Modeling"]
        }
    }
    return professional_data

class DatabricksRAG:
    """RAG implementation using Databricks native capabilities"""
    
    def __init__(self):
        self.client = get_databricks_client()
        self.embedding_model = load_embedding_model()
        self.professional_data = load_professional_data()
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
        
        if self.embedding_model == "databricks" and self.client:
            # Use Databricks embedding endpoint
            try:
                response = self.client.predict(
                    endpoint=Config.EMBEDDING_MODEL,
                    inputs={"input": self.documents}
                )
                embeddings = response["data"]
            except Exception as e:
                st.warning(f"Databricks embeddings failed, using fallback: {e}")
                # Fallback to sentence-transformers
                model = SentenceTransformer(Config.FALLBACK_EMBEDDING_MODEL)
                embeddings = model.encode(self.documents)
        else:
            # Use sentence-transformers
            embeddings = self.embedding_model.encode(self.documents)
        
        return embeddings
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for a query using similarity search"""
        # Create query embedding
        if self.embedding_model == "databricks" and self.client:
            try:
                query_response = self.client.predict(
                    endpoint=Config.EMBEDDING_MODEL,
                    inputs={"input": [query]}
                )
                query_embedding = np.array(query_response["data"][0])
            except:
                # Fallback
                model = SentenceTransformer(Config.FALLBACK_EMBEDDING_MODEL)
                query_embedding = model.encode([query])[0]
        else:
            query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_docs = []
        for idx in top_indices:
            relevant_docs.append(self.documents[idx])
        
        return "\n\n".join(relevant_docs)

class DatabricksChatBot:
    """Chatbot using Databricks foundation models"""
    
    def __init__(self):
        self.client = get_databricks_client()
        self.rag = DatabricksRAG()
        
    def generate_response(self, user_query: str, chat_history: List[Dict]) -> str:
        """Generate response using Databricks foundation model"""
        
        if not self.client:
            return "I'm sorry, but I'm having trouble connecting to Databricks services right now."
        
        # Retrieve relevant professional context
        context = self.rag.retrieve_context(user_query)
        
        # Create system message
        system_message = f"""You are a professional AI assistant representing a skilled data scientist and ML engineer with extensive Databricks experience. 
        
        Your role is to discuss their professional background, technical expertise, and project experience based on the provided context.
        
        Professional Context:
        {context}
        
        Guidelines:
        - Be enthusiastic and professional when discussing their experience
        - Highlight their Databricks expertise and data engineering skills
        - Reference specific projects and achievements from the context
        - If asked about technologies they haven't used, be honest but mention related experience
        - Keep responses conversational but informative
        - Focus on their unique value proposition as a Databricks expert
        """
        
        # Prepare messages for foundation model
        messages = [{"role": "system", "content": system_message}]
        
        # Add recent chat history (limit to stay within context window)
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
                    "max_tokens": 500,
                    "temperature": 0.7
                }
            )
            
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request. Please try again. Error: {str(e)}"

# Streamlit App
def streamlit_app():
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