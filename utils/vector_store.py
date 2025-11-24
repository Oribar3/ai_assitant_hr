"""Vector store utilities for semantic search and similarity matching."""

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger


class VectorStoreManager:
    """Manages vector embeddings and similarity computations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store manager.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into vector embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Vector embedding as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into vector embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of vector embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True)
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        vec1 = self.encode_text(text1)
        vec2 = self.encode_text(text2)
        return self.cosine_similarity(vec1, vec2)
    
    def find_matches(
        self,
        query_items: List[str],
        candidate_items: List[str],
        threshold: float = 0.6
    ) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        """
        Find matching items between two lists using semantic similarity.
        
        Args:
            query_items: List of items to match (e.g., required skills)
            candidate_items: List of candidate items (e.g., candidate's skills)
            threshold: Minimum similarity threshold for a match
            
        Returns:
            Tuple of (matched_items, all_matches_with_scores)
        """
        if not query_items or not candidate_items:
            return [], []
        
        # Encode all items
        query_embeddings = self.encode_texts(query_items)
        candidate_embeddings = self.encode_texts(candidate_items)
        
        matched = []
        all_matches = []
        
        for i, query_item in enumerate(query_items):
            query_vec = query_embeddings[i]
            best_match = None
            best_score = 0.0
            
            for j, candidate_item in enumerate(candidate_items):
                candidate_vec = candidate_embeddings[j]
                score = self.cosine_similarity(query_vec, candidate_vec)
                
                if score > best_score:
                    best_score = score
                    best_match = candidate_item
            
            if best_score >= threshold and best_match:
                matched.append(query_item)
                all_matches.append((query_item, best_match, best_score))
        
        return matched, all_matches
    
    def calculate_skill_match_score(
        self,
        required_skills: List[str],
        candidate_skills: List[str],
        preferred_skills: List[str] = None,
        required_weight: float = 1.5,
        preferred_weight: float = 1.0
    ) -> Tuple[float, dict]:
        """
        Calculate overall skill match score.
        
        Args:
            required_skills: List of required skills
            candidate_skills: List of candidate's skills
            preferred_skills: List of preferred skills (optional)
            required_weight: Weight for required skills
            preferred_weight: Weight for preferred skills
            
        Returns:
            Tuple of (overall_score, details_dict)
        """
        if not required_skills:
            return 0.0, {}
        
        # Find matches for required skills
        matched_required, required_matches = self.find_matches(
            required_skills, candidate_skills
        )
        
        required_match_rate = len(matched_required) / len(required_skills)
        score = required_match_rate * required_weight
        
        details = {
            "matched_required": matched_required,
            "missing_required": [
                skill for skill in required_skills 
                if skill not in matched_required
            ],
            "required_match_rate": required_match_rate
        }
        
        # Find matches for preferred skills if provided
        if preferred_skills:
            matched_preferred, preferred_matches = self.find_matches(
                preferred_skills, candidate_skills
            )
            
            preferred_match_rate = len(matched_preferred) / len(preferred_skills)
            score += preferred_match_rate * preferred_weight
            
            details["matched_preferred"] = matched_preferred
            details["preferred_match_rate"] = preferred_match_rate
            
            # Normalize score
            total_weight = required_weight + preferred_weight
            score = (score / total_weight) * 100
        else:
            # Normalize score without preferred skills
            score = (score / required_weight) * 100
        
        return min(score, 100.0), details
    
    def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Perform semantic search on documents.
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of top results to return
            
        Returns:
            List of (document_index, similarity_score) tuples
        """
        if not documents:
            return []
        
        query_vec = self.encode_text(query)
        doc_vecs = self.encode_texts(documents)
        
        similarities = []
        for i, doc_vec in enumerate(doc_vecs):
            score = self.cosine_similarity(query_vec, doc_vec)
            similarities.append((i, score))
        
        # Sort by score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
