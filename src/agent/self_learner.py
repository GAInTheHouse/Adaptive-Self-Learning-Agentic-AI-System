"""
Self-Learning Module - Week 2 (Team Member 1)
Lightweight in-memory error tracking for agent integration.
Note: Data persistence and management handled by Team Member 2.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelfLearner:
    """
    Lightweight self-learning component for agent integration.
    Tracks errors in-memory for agent decision-making.
    Note: Data persistence/management is handled by the data management layer.
    """
    
    def __init__(self):
        """
        Initialize self-learner with in-memory tracking only.
        """
        # In-memory error tracking (data persistence handled by data layer)
        self.error_patterns = defaultdict(list)  # error_type -> [instances]
        self.correction_history = []  # List of corrections made
        self.error_statistics = defaultdict(int)  # error_type -> count
        self.feedback_history = []  # User feedback (in-memory only)
        
        logger.info("Self-learner initialized (in-memory tracking only)")
    
    def record_error(
        self,
        error_type: str,
        transcript: str,
        context: Optional[Dict] = None,
        correction: Optional[str] = None
    ):
        """
        Record an error for in-memory tracking.
        Note: Data persistence handled by data management layer.
        
        Args:
            error_type: Type of error detected
            transcript: The transcript with error
            context: Additional context (audio length, model confidence, etc.)
            correction: Corrected transcript if available
        """
        error_instance = {
            'error_type': error_type,
            'transcript': transcript,
            'context': context or {},
            'correction': correction,
            'timestamp': datetime.now().isoformat()
        }
        
        self.error_patterns[error_type].append(error_instance)
        self.error_statistics[error_type] += 1
        
        if correction:
            self.correction_history.append({
                'original': transcript,
                'corrected': correction,
                'error_type': error_type,
                'timestamp': datetime.now().isoformat()
            })
    
    def record_feedback(
        self,
        transcript_id: str,
        user_feedback: str,
        is_correct: bool,
        corrected_transcript: Optional[str] = None
    ):
        """
        Record user feedback for in-memory tracking.
        Note: Data persistence handled by data management layer.
        
        Args:
            transcript_id: Identifier for the transcript
            user_feedback: User's feedback text
            is_correct: Whether the transcript was correct
            corrected_transcript: Corrected version if not correct
        """
        feedback_entry = {
            'transcript_id': transcript_id,
            'user_feedback': user_feedback,
            'is_correct': is_correct,
            'corrected_transcript': corrected_transcript,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback_entry)
        logger.info(f"Recorded feedback: {'correct' if is_correct else 'incorrect'}")
    
    def get_error_statistics(self) -> Dict:
        """
        Get statistics about learned errors.
        
        Returns:
            Dictionary with error statistics
        """
        total_errors = sum(self.error_statistics.values())
        
        return {
            'total_errors': total_errors,
            'error_type_counts': dict(self.error_statistics),
            'error_type_distribution': {
                error_type: count / total_errors if total_errors > 0 else 0
                for error_type, count in self.error_statistics.items()
            },
            'corrections_made': len(self.correction_history),
            'feedback_count': len(self.feedback_history)
        }
    
    def get_common_error_patterns(self, error_type: Optional[str] = None) -> List[Dict]:
        """
        Get common error patterns for a specific error type or all types.
        
        Args:
            error_type: Specific error type to analyze, or None for all
        
        Returns:
            List of error pattern dictionaries
        """
        if error_type:
            patterns = self.error_patterns.get(error_type, [])
        else:
            patterns = []
            for patterns_list in self.error_patterns.values():
                patterns.extend(patterns_list)
        
        # Sort by frequency (most common first)
        pattern_counts = defaultdict(int)
        for pattern in patterns:
            key = pattern.get('transcript', '')[:50]  # Use first 50 chars as key
            pattern_counts[key] += 1
        
        return sorted(
            [
                {
                    'pattern': pattern,
                    'frequency': count,
                    'error_type': error_type or 'all'
                }
                for pattern, count in pattern_counts.items()
            ],
            key=lambda x: x['frequency'],
            reverse=True
        )[:10]  # Top 10 patterns
    
    def get_in_memory_data(self) -> Dict:
        """
        Get all in-memory learning data for external persistence.
        Note: This data should be persisted by the data management layer.
        
        Returns:
            Dictionary with all learning data
        """
        return {
            'error_patterns': dict(self.error_patterns),
            'correction_history': self.correction_history,
            'error_statistics': dict(self.error_statistics),
            'feedback_history': self.feedback_history,
            'last_updated': datetime.now().isoformat()
        }
    
    def load_from_data(self, data: Dict):
        """
        Load learning data from external source (data management layer).
        
        Args:
            data: Dictionary with learning data
        """
        self.error_patterns = defaultdict(list, data.get('error_patterns', {}))
        self.correction_history = data.get('correction_history', [])
        self.error_statistics = defaultdict(int, data.get('error_statistics', {}))
        self.feedback_history = data.get('feedback_history', [])
        logger.info("Loaded learning data from external source")

