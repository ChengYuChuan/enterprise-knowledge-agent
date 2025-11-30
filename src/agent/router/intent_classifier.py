"""
Intent classifier for query routing.

This module provides rule-based intent classification without requiring
an LLM. For Phase 3, we use heuristics and patterns. In Phase 4, this
can be enhanced with LLM-based classification.
"""

import re
from typing import Optional

from .base import QueryAnalysis, QueryIntent


class IntentClassifier:
    """
    Rule-based intent classifier.
    
    Uses pattern matching and heuristics to classify user queries.
    This is a simplified version for Phase 3. In Phase 4, we can
    enhance this with LLM-based classification for better accuracy.
    
    Classification rules:
    1. Greetings/chitchat: "hello", "hi", "thanks", "bye"
    2. Admin requests: "how many", "list", "show all", "statistics"
    3. Comparisons: "compare", "difference between", "vs"
    4. Complex reasoning: "why", "explain", "analyze"
    5. Factual lookup: "what is", "who is", "when is"
    6. Knowledge query: Default for domain-specific questions
    """
    
    # Pattern definitions
    GREETING_PATTERNS = [
        r'\b(hello|hi|hey|greetings)\b',
        r'\b(thanks|thank you|appreciate)\b',
        r'\b(bye|goodbye|see you)\b',
        r'^(how are you|good morning|good afternoon)',
    ]
    
    ADMIN_PATTERNS = [
        r'\b(how many|count|total|number of)\b',
        r'\b(list|show|display) (all|documents|files)\b',
        r'\b(stats|statistics|status|info)\b',
        r'what (documents|files) (do we have|are available)',
    ]
    
    COMPARISON_PATTERNS = [
        r'\b(compare|comparison|versus|vs)\b',
        r'\b(difference between|differ from)\b',
        r'\b(better|worse|more|less) than\b',
        r'(how does .* compare)',
    ]
    
    COMPLEX_REASONING_PATTERNS = [
        r'\b(why|how come|reason for)\b',
        r'\b(explain|analyze|elaborate)\b',
        r'\b(what if|suppose|assuming)\b',
        r'(multiple|several|various) .* (and|or)',
    ]
    
    FACTUAL_LOOKUP_PATTERNS = [
        r'^(what is|who is|when is|where is)',
        r'^(define|definition of)\b',
        r'\b(specific|exact|precisely)\b',
    ]
    
    def __init__(self) -> None:
        """Initialize the classifier."""
        # Compile patterns for efficiency
        self.greeting_regex = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self.admin_regex = [re.compile(p, re.IGNORECASE) for p in self.ADMIN_PATTERNS]
        self.comparison_regex = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self.complex_regex = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_REASONING_PATTERNS]
        self.factual_regex = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_LOOKUP_PATTERNS]
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify the intent of a user query.
        
        Args:
            query: User query string.
            
        Returns:
            QueryIntent: Classified intent.
        """
        query_lower = query.lower().strip()
        
        # Check for chitchat/greetings first
        if self._matches_any_pattern(query_lower, self.greeting_regex):
            return QueryIntent.CHITCHAT
        
        # Check for admin requests
        if self._matches_any_pattern(query_lower, self.admin_regex):
            return QueryIntent.ADMIN_REQUEST
        
        # Check for comparisons
        if self._matches_any_pattern(query_lower, self.comparison_regex):
            return QueryIntent.COMPARISON
        
        # Check for complex reasoning
        if self._matches_any_pattern(query_lower, self.complex_regex):
            return QueryIntent.COMPLEX_REASONING
        
        # Check for factual lookups
        if self._matches_any_pattern(query_lower, self.factual_regex):
            return QueryIntent.FACTUAL_LOOKUP
        
        # Default: knowledge query
        # If it contains domain-specific terms, it's likely a knowledge query
        if len(query_lower.split()) >= 3:
            return QueryIntent.KNOWLEDGE_QUERY
        
        # Very short queries are unclear
        if len(query_lower.split()) <= 2:
            return QueryIntent.UNCLEAR
        
        return QueryIntent.KNOWLEDGE_QUERY
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform detailed analysis of a query.
        
        Args:
            query: User query string.
            
        Returns:
            QueryAnalysis: Detailed query analysis.
        """
        query_lower = query.lower().strip()
        
        # Extract keywords (remove stop words)
        keywords = self._extract_keywords(query)
        
        # Determine question type
        question_type = self._get_question_type(query)
        
        # Extract entities (simplified - no NER model yet)
        entities = self._extract_entities(query)
        
        # Check if KB is needed
        requires_kb = not self._matches_any_pattern(query_lower, self.greeting_regex)
        
        # Check if greeting
        is_greeting = self._matches_any_pattern(query_lower, self.greeting_regex)
        
        # Estimate complexity
        complexity = self._estimate_complexity(query, keywords)
        
        return QueryAnalysis(
            query=query,
            keywords=keywords,
            question_type=question_type,
            entities=entities,
            requires_knowledge_base=requires_kb,
            is_greeting=is_greeting,
            complexity_score=complexity,
        )
    
    def _matches_any_pattern(self, text: str, patterns: list[re.Pattern]) -> bool:
        """
        Check if text matches any of the given patterns.
        
        Args:
            text: Text to check.
            patterns: List of compiled regex patterns.
            
        Returns:
            bool: True if any pattern matches.
        """
        return any(pattern.search(text) for pattern in patterns)
    
    def _extract_keywords(self, query: str) -> list[str]:
        """
        Extract keywords from query (remove common stop words).
        
        Args:
            query: User query.
            
        Returns:
            list[str]: List of keywords.
        """
        # Simple stop words list
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'of', 'to',
            'for', 'in', 'on', 'at', 'by', 'with', 'from', 'about', 'as',
            'what', 'when', 'where', 'who', 'how', 'why', 'which',
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _get_question_type(self, query: str) -> Optional[str]:
        """
        Identify question type (what, how, when, etc.).
        
        Args:
            query: User query.
            
        Returns:
            Optional[str]: Question type or None.
        """
        query_lower = query.lower().strip()
        
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which']
        
        for qword in question_words:
            if query_lower.startswith(qword):
                return qword
        
        return None
    
    def _extract_entities(self, query: str) -> list[str]:
        """
        Extract named entities (simplified).
        
        In Phase 4, this can be enhanced with proper NER.
        
        Args:
            query: User query.
            
        Returns:
            list[str]: List of potential entities.
        """
        # Very simple: extract capitalized words (likely proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Also look for quoted terms
        quoted = re.findall(r'"([^"]+)"', query)
        
        return list(set(entities + quoted))
    
    def _estimate_complexity(self, query: str, keywords: list[str]) -> float:
        """
        Estimate query complexity (0-1 scale).
        
        Factors:
        - Number of keywords
        - Presence of complex patterns
        - Query length
        
        Args:
            query: User query.
            keywords: Extracted keywords.
            
        Returns:
            float: Complexity score (0-1).
        """
        complexity = 0.0
        
        # Base complexity from keyword count
        complexity += min(len(keywords) / 10, 0.3)
        
        # Add for complex reasoning patterns
        if self._matches_any_pattern(query.lower(), self.complex_regex):
            complexity += 0.3
        
        # Add for comparisons
        if self._matches_any_pattern(query.lower(), self.comparison_regex):
            complexity += 0.2
        
        # Add for query length
        word_count = len(query.split())
        complexity += min(word_count / 20, 0.2)
        
        return min(complexity, 1.0)