"""
Adaptive Scheduling Algorithm - Week 3 (Team Member 3)
Dynamic threshold adjustment and performance-aware fine-tuning scheduling.
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from dataclasses import dataclass, field
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track performance metrics over time"""
    timestamp: datetime
    error_count: int
    accuracy: float  # Model accuracy (1 - error_rate)
    wer: Optional[float] = None  # Word Error Rate
    cer: Optional[float] = None  # Character Error Rate
    inference_time: float = 0.0
    cost_per_inference: float = 0.0  # Computational cost estimate
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_count': self.error_count,
            'accuracy': self.accuracy,
            'wer': self.wer,
            'cer': self.cer,
            'inference_time': self.inference_time,
            'cost_per_inference': self.cost_per_inference
        }


@dataclass
class FineTuningEvent:
    """Record of a fine-tuning event"""
    timestamp: datetime
    threshold_n: int  # Threshold n used for triggering
    samples_used: int  # Number of error samples used
    validation_accuracy_before: float
    validation_accuracy_after: float
    accuracy_gain: float
    training_cost: float  # Computational cost of fine-tuning
    overfitting_detected: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'threshold_n': self.threshold_n,
            'samples_used': self.samples_used,
            'validation_accuracy_before': self.validation_accuracy_before,
            'validation_accuracy_after': self.validation_accuracy_after,
            'accuracy_gain': self.accuracy_gain,
            'training_cost': self.training_cost,
            'overfitting_detected': self.overfitting_detected
        }


class AdaptiveScheduler:
    """
    Adaptive scheduling mechanism that dynamically adjusts threshold n
    based on performance metrics and cost-efficiency considerations.
    """
    
    def __init__(
        self,
        initial_threshold_n: int = 100,
        min_threshold_n: int = 50,
        max_threshold_n: int = 1000,
        performance_window_size: int = 20,
        accuracy_gain_threshold: float = 0.01,  # Minimum accuracy gain to consider worthwhile
        diminishing_gain_threshold: float = 0.005,  # Below this, gains are diminishing
        cost_efficiency_weight: float = 0.3,  # Weight for cost in scheduling decisions
        validation_split: float = 0.2,  # Fraction of data for validation
        overfitting_threshold: float = 0.1,  # Max difference between train and validation accuracy
        history_path: Optional[str] = None
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            initial_threshold_n: Starting threshold for number of errors before fine-tuning
            min_threshold_n: Minimum allowed threshold n
            max_threshold_n: Maximum allowed threshold n
            performance_window_size: Number of recent metrics to consider
            accuracy_gain_threshold: Minimum accuracy gain to consider fine-tuning worthwhile
            diminishing_gain_threshold: Threshold below which gains are considered diminishing
            cost_efficiency_weight: Weight for cost considerations (0-1)
            validation_split: Fraction of data to use for validation
            overfitting_threshold: Max allowed difference between train/val accuracy
            history_path: Path to save/load scheduler history
        """
        # Threshold management
        self.current_threshold_n = initial_threshold_n
        self.min_threshold_n = min_threshold_n
        self.max_threshold_n = max_threshold_n
        
        # Performance tracking
        self.performance_window_size = performance_window_size
        self.performance_history: deque = deque(maxlen=performance_window_size)
        self.fine_tuning_history: List[FineTuningEvent] = []
        
        # Configuration
        self.accuracy_gain_threshold = accuracy_gain_threshold
        self.diminishing_gain_threshold = diminishing_gain_threshold
        self.cost_efficiency_weight = cost_efficiency_weight
        self.validation_split = validation_split
        self.overfitting_threshold = overfitting_threshold
        
        # Cost tracking
        self.total_training_cost = 0.0
        self.total_inference_cost = 0.0
        self.cost_per_sample_estimate = 0.001  # Estimated cost per inference sample
        
        # Error sample collection
        self.error_samples_collected = 0
        
        # History persistence
        self.history_path = history_path
        if history_path:
            self._load_history()
        
        logger.info(f"Adaptive scheduler initialized with threshold_n={initial_threshold_n}")
    
    def should_trigger_fine_tuning(self) -> Tuple[bool, Dict]:
        """
        Determine if fine-tuning should be triggered based on current threshold n.
        
        Returns:
            Tuple of (should_trigger, decision_info)
        """
        should_trigger = self.error_samples_collected >= self.current_threshold_n
        
        decision_info = {
            'should_trigger': should_trigger,
            'current_threshold_n': self.current_threshold_n,
            'samples_collected': self.error_samples_collected,
            'samples_remaining': max(0, self.current_threshold_n - self.error_samples_collected)
        }
        
        return should_trigger, decision_info
    
    def record_performance(
        self,
        error_count: int,
        accuracy: float,
        wer: Optional[float] = None,
        cer: Optional[float] = None,
        inference_time: float = 0.0,
        cost_per_inference: Optional[float] = None
    ):
        """
        Record performance metrics for adaptive scheduling.
        
        Args:
            error_count: Number of errors detected
            accuracy: Model accuracy (1 - error_rate)
            wer: Word Error Rate (optional)
            cer: Character Error Rate (optional)
            inference_time: Time taken for inference (seconds)
            cost_per_inference: Estimated cost per inference (optional)
        """
        cost = cost_per_inference or (self.cost_per_sample_estimate * inference_time)
        
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            error_count=error_count,
            accuracy=accuracy,
            wer=wer,
            cer=cer,
            inference_time=inference_time,
            cost_per_inference=cost
        )
        
        self.performance_history.append(metric)
        self.total_inference_cost += cost
        
        logger.debug(f"Recorded performance: accuracy={accuracy:.4f}, errors={error_count}")
    
    def record_error_sample(self, count: int = 1):
        """
        Record that error samples have been collected.
        
        Args:
            count: Number of error samples to add
        """
        self.error_samples_collected += count
        logger.debug(f"Error samples collected: {self.error_samples_collected}/{self.current_threshold_n}")
    
    def adjust_threshold_n(self, fine_tuning_result: Optional[Dict] = None):
        """
        Dynamically adjust threshold n based on performance trends and fine-tuning results.
        
        Args:
            fine_tuning_result: Dictionary with fine-tuning results if available
        """
        if len(self.performance_history) < 5:
            # Not enough data to make adjustments
            return
        
        # Analyze recent performance trends
        recent_metrics = list(self.performance_history)[-10:]
        accuracy_trend = self._calculate_accuracy_trend(recent_metrics)
        
        # Check if accuracy gains are diminishing
        diminishing_gains = self._detect_diminishing_gains()
        
        # Calculate cost efficiency
        cost_efficiency = self._calculate_cost_efficiency()
        
        # Adjust threshold based on multiple factors
        adjustment_factor = 1.0
        
        # Factor 1: Diminishing gains -> increase threshold n
        if diminishing_gains:
            adjustment_factor *= 1.2  # Increase threshold by 20%
            logger.info("Diminishing accuracy gains detected, increasing threshold_n")
        
        # Factor 2: Poor cost efficiency -> increase threshold n
        if cost_efficiency < 0.5:
            adjustment_factor *= 1.15  # Increase threshold by 15%
            logger.info("Poor cost efficiency detected, increasing threshold_n")
        
        # Factor 3: Recent fine-tuning result
        if fine_tuning_result:
            accuracy_gain = fine_tuning_result.get('accuracy_gain', 0.0)
            if accuracy_gain < self.diminishing_gain_threshold:
                adjustment_factor *= 1.3  # Significant increase if gains are very small
                logger.info(f"Small accuracy gain ({accuracy_gain:.4f}), significantly increasing threshold_n")
            elif accuracy_gain > self.accuracy_gain_threshold:
                adjustment_factor *= 0.9  # Slight decrease if gains are good
                logger.info(f"Good accuracy gain ({accuracy_gain:.4f}), slightly decreasing threshold_n")
        
        # Factor 4: Accuracy trend
        if accuracy_trend < -0.01:  # Accuracy declining
            adjustment_factor *= 0.95  # Slight decrease to fine-tune more frequently
            logger.info("Accuracy declining, slightly decreasing threshold_n")
        elif accuracy_trend > 0.01:  # Accuracy improving
            adjustment_factor *= 1.1  # Increase threshold as model is improving
            logger.info("Accuracy improving, increasing threshold_n")
        
        # Apply adjustment with bounds
        new_threshold = int(self.current_threshold_n * adjustment_factor)
        new_threshold = max(self.min_threshold_n, min(self.max_threshold_n, new_threshold))
        
        if new_threshold != self.current_threshold_n:
            old_threshold = self.current_threshold_n
            self.current_threshold_n = new_threshold
            logger.info(f"Adjusted threshold_n: {old_threshold} -> {new_threshold} (factor: {adjustment_factor:.2f})")
    
    def _calculate_accuracy_trend(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate the trend in accuracy over recent metrics."""
        if len(metrics) < 2:
            return 0.0
        
        accuracies = [m.accuracy for m in metrics]
        # Simple linear trend
        x = np.arange(len(accuracies))
        trend = np.polyfit(x, accuracies, 1)[0]  # Linear coefficient
        
        return trend
    
    def _detect_diminishing_gains(self) -> bool:
        """Detect if accuracy gains are diminishing based on fine-tuning history."""
        if len(self.fine_tuning_history) < 2:
            return False
        
        # Check last few fine-tuning events
        recent_events = self.fine_tuning_history[-3:]
        gains = [event.accuracy_gain for event in recent_events]
        
        if len(gains) < 2:
            return False
        
        # Check if gains are decreasing
        if gains[-1] < self.diminishing_gain_threshold:
            return True
        
        # Check if trend is decreasing
        if len(gains) >= 3:
            trend = np.polyfit(range(len(gains)), gains, 1)[0]
            if trend < 0 and gains[-1] < self.accuracy_gain_threshold:
                return True
        
        return False
    
    def _calculate_cost_efficiency(self) -> float:
        """Calculate cost efficiency score (0-1, higher is better)."""
        if len(self.fine_tuning_history) == 0:
            return 1.0  # No fine-tuning yet, assume efficient
        
        # Calculate average accuracy gain per unit cost
        total_gain = sum(event.accuracy_gain for event in self.fine_tuning_history)
        total_cost = sum(event.training_cost for event in self.fine_tuning_history)
        
        if total_cost == 0:
            return 1.0
        
        efficiency = total_gain / total_cost
        
        # Normalize to 0-1 scale (assuming efficiency > 0.1 is good)
        normalized_efficiency = min(1.0, efficiency / 0.1)
        
        return normalized_efficiency
    
    def record_fine_tuning_event(
        self,
        samples_used: int,
        validation_accuracy_before: float,
        validation_accuracy_after: float,
        training_cost: float,
        overfitting_detected: bool = False
    ):
        """
        Record a fine-tuning event and adjust threshold accordingly.
        
        Args:
            samples_used: Number of error samples used for fine-tuning
            validation_accuracy_before: Validation accuracy before fine-tuning
            validation_accuracy_after: Validation accuracy after fine-tuning
            training_cost: Computational cost of fine-tuning
            overfitting_detected: Whether overfitting was detected
        """
        accuracy_gain = validation_accuracy_after - validation_accuracy_before
        
        event = FineTuningEvent(
            timestamp=datetime.now(),
            threshold_n=self.current_threshold_n,
            samples_used=samples_used,
            validation_accuracy_before=validation_accuracy_before,
            validation_accuracy_after=validation_accuracy_after,
            accuracy_gain=accuracy_gain,
            training_cost=training_cost,
            overfitting_detected=overfitting_detected
        )
        
        self.fine_tuning_history.append(event)
        self.total_training_cost += training_cost
        self.error_samples_collected = 0  # Reset counter after fine-tuning
        
        # Adjust threshold based on this fine-tuning result
        fine_tuning_result = {
            'accuracy_gain': accuracy_gain,
            'overfitting_detected': overfitting_detected
        }
        self.adjust_threshold_n(fine_tuning_result)
        
        logger.info(
            f"Recorded fine-tuning event: accuracy_gain={accuracy_gain:.4f}, "
            f"cost={training_cost:.2f}, overfitting={overfitting_detected}"
        )
    
    def check_overfitting(
        self,
        train_accuracy: float,
        validation_accuracy: float
    ) -> Tuple[bool, Dict]:
        """
        Check for overfitting by comparing train and validation accuracy.
        
        Args:
            train_accuracy: Training accuracy
            validation_accuracy: Validation accuracy
        
        Returns:
            Tuple of (is_overfitting, overfitting_info)
        """
        accuracy_gap = train_accuracy - validation_accuracy
        is_overfitting = accuracy_gap > self.overfitting_threshold
        
        overfitting_info = {
            'is_overfitting': is_overfitting,
            'train_accuracy': train_accuracy,
            'validation_accuracy': validation_accuracy,
            'accuracy_gap': accuracy_gap,
            'threshold': self.overfitting_threshold
        }
        
        if is_overfitting:
            logger.warning(
                f"Overfitting detected: train_acc={train_accuracy:.4f}, "
                f"val_acc={validation_accuracy:.4f}, gap={accuracy_gap:.4f}"
            )
        
        return is_overfitting, overfitting_info
    
    def get_scheduler_stats(self) -> Dict:
        """Get comprehensive scheduler statistics."""
        recent_accuracy = None
        accuracy_trend = 0.0
        
        if len(self.performance_history) > 0:
            recent_metrics = list(self.performance_history)[-5:]
            recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
            accuracy_trend = self._calculate_accuracy_trend(recent_metrics)
        
        avg_accuracy_gain = 0.0
        if len(self.fine_tuning_history) > 0:
            avg_accuracy_gain = np.mean([e.accuracy_gain for e in self.fine_tuning_history])
        
        return {
            'current_threshold_n': self.current_threshold_n,
            'error_samples_collected': self.error_samples_collected,
            'samples_until_fine_tuning': max(0, self.current_threshold_n - self.error_samples_collected),
            'recent_accuracy': recent_accuracy,
            'accuracy_trend': accuracy_trend,
            'diminishing_gains_detected': self._detect_diminishing_gains(),
            'cost_efficiency': self._calculate_cost_efficiency(),
            'total_fine_tuning_events': len(self.fine_tuning_history),
            'average_accuracy_gain': avg_accuracy_gain,
            'total_training_cost': self.total_training_cost,
            'total_inference_cost': self.total_inference_cost,
            'total_cost': self.total_training_cost + self.total_inference_cost
        }
    
    def _load_history(self):
        """Load scheduler history from file."""
        if not self.history_path:
            return
        
        history_file = Path(self.history_path)
        if not history_file.exists():
            return
        
        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
            
            # Load threshold
            self.current_threshold_n = data.get('current_threshold_n', self.current_threshold_n)
            self.error_samples_collected = data.get('error_samples_collected', 0)
            
            # Load fine-tuning history
            self.fine_tuning_history = [
                FineTuningEvent(**event_data)
                for event_data in data.get('fine_tuning_history', [])
            ]
            
            # Load performance history (limited to window size)
            performance_data = data.get('performance_history', [])
            for metric_data in performance_data[-self.performance_window_size:]:
                metric = PerformanceMetrics(
                    timestamp=datetime.fromisoformat(metric_data['timestamp']),
                    error_count=metric_data['error_count'],
                    accuracy=metric_data['accuracy'],
                    wer=metric_data.get('wer'),
                    cer=metric_data.get('cer'),
                    inference_time=metric_data.get('inference_time', 0.0),
                    cost_per_inference=metric_data.get('cost_per_inference', 0.0)
                )
                self.performance_history.append(metric)
            
            logger.info(f"Loaded scheduler history from {self.history_path}")
        except Exception as e:
            logger.warning(f"Failed to load scheduler history: {e}")
    
    def save_history(self):
        """Save scheduler history to file."""
        if not self.history_path:
            return
        
        try:
            history_file = Path(self.history_path)
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'current_threshold_n': self.current_threshold_n,
                'error_samples_collected': self.error_samples_collected,
                'fine_tuning_history': [event.to_dict() for event in self.fine_tuning_history],
                'performance_history': [metric.to_dict() for metric in self.performance_history]
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved scheduler history to {self.history_path}")
        except Exception as e:
            logger.warning(f"Failed to save scheduler history: {e}")
