import time
import json
import requests
from PySide6.QtCore import QThread, Signal

class LMStudioChatThread(QThread):
    """Thread for handling LM Studio chat interactions"""
    response_ready = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.queue = []
        self.model_name = None
        self.json_data = {}
        self.connection_error = False
        self.lm_studio_url = "http://localhost:1234/v1"
    
    def set_model(self, model_name):
        """Set LM Studio model"""
        self.model_name = model_name
        
    def set_json_data(self, data):
        """Set JSON data for quick responses"""
        self.json_data = data
    
    def add_query(self, query, context=None):
        """Add query to processing queue"""
        self.queue.append((query, context))
    
    def run(self):
        while self.running:
            if self.queue:
                query, context = self.queue.pop(0)
                try:
                    # First try to get quick response from JSON
                    quick_response = self.get_quick_response(query, context)
                    if quick_response:
                        self.response_ready.emit(quick_response)
                        
                    # Then get detailed response from LM Studio
                    detailed_response = self.get_detailed_response(query, context)
                    if detailed_response:
                        self.response_ready.emit(detailed_response)
                        
                except Exception as e:
                    self.error_occurred.emit(str(e))
                    
            time.sleep(0.1)
    
    def get_quick_response(self, query, context=None):
        """Get quick response from JSON data"""
        if not context or not self.json_data:
            return None
            
        # Check if context exists in JSON data
        if context in self.json_data:
            info = self.json_data[context]
            
            # Analyze query type
            query_lower = query.lower()
            
            if "ne" in query_lower or "nedir" in query_lower:
                return info.get("definition", "")
            elif "özellik" in query_lower:
                return info.get("features", "")
            elif "kullanım" in query_lower or "nasıl" in query_lower:
                return info.get("usage", "")
            elif "ilginç" in query_lower:
                return info.get("interesting_facts", "")
                
        return None
    
    def get_custom_quick_response(self, query):
        """Get quick response for custom queries"""
        query_lower = query.lower()
        
        # Common questions
        if "merhaba" in query_lower or "selam" in query_lower:
            return "Merhaba! Size nasıl yardımcı olabilirim?"
        elif "teşekkür" in query_lower:
            return "Rica ederim! Başka bir sorunuz var mı?"
        elif "yardım" in query_lower:
            return "Size nesneler hakkında bilgi verebilir, tanımlayabilir ve özelliklerini açıklayabilirim."
            
        return None
    
    def get_detailed_response(self, query, context=None):
        """Get detailed response from LM Studio"""
        if self.connection_error:
            return None
            
        try:
            # Prepare prompt
            prompt = f"Context: {context}\nQuestion: {query}" if context else query
            
            # Make request to LM Studio
            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                self.connection_error = True
                return None
                
        except Exception as e:
            self.connection_error = True
            return None
    
    def stop(self):
        """Stop chat thread"""
        self.running = False
        self.wait() 