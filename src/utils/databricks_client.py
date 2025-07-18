import os
import mlflow
import mlflow.deployments
from typing import Optional, Dict, Any
import streamlit as st
from config import databricks_config

class DatabricksClient:
    """Wrapper for Databricks MLflow client with local development support"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Databricks client with proper authentication"""
        try:
            # Set environment variables for authentication
            if databricks_config.host:
                os.environ["DATABRICKS_HOST"] = databricks_config.host
            if databricks_config.token:
                os.environ["DATABRICKS_TOKEN"] = databricks_config.token
            
            # Initialize MLflow client
            self.client = mlflow.deployments.get_deploy_client("databricks")
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            print(f"Warning: Could not initialize Databricks client: {e}")
            self.client = None
    
    def _test_connection(self):
        """Test the Databricks connection"""
        if self.client:
            try:
                # Try to list endpoints to verify connection
                endpoints = self.client.list_endpoints()
                print(f"Successfully connected to Databricks. Found {len(endpoints)} endpoints.")
            except Exception as e:
                print(f"Connection test failed: {e}")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if Databricks client is available"""
        return self.client is not None
    
    def predict(self, endpoint: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction with error handling"""
        if not self.client:
            raise Exception("Databricks client not available")
        
        try:
            return self.client.predict(endpoint=endpoint, inputs=inputs)
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")
    
    def list_endpoints(self):
        """List available endpoints"""
        if not self.client:
            return []
        
        try:
            return self.client.list_endpoints()
        except Exception as e:
            print(f"Error listing endpoints: {e}")
            return []

# Global client instance
databricks_client = DatabricksClient()