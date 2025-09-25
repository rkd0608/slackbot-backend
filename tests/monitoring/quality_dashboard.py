"""Quality monitoring dashboard for AI response evaluation."""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from loguru import logger

from ..evaluators.factual_accuracy import factual_accuracy_evaluator
from ..evaluators.completeness_evaluator import completeness_evaluator
from ..evaluators.hallucination_detector import hallucination_detector
from ..evaluators.response_utility_scorer import response_utility_scorer


@dataclass
class QualityMetrics:
    """Quality metrics for a single evaluation."""
    timestamp: datetime
    response_id: str
    query: str
    response: str
    factual_accuracy: Dict[str, float]
    completeness: Dict[str, float]
    hallucination: Dict[str, Any]
    utility: Dict[str, float]
    overall_score: float


@dataclass
class QualityThresholds:
    """Quality thresholds for evaluation."""
    factual_accuracy_min: float = 0.8
    completeness_min: float = 0.7
    hallucination_max: float = 0.3
    utility_min: float = 0.75
    overall_min: float = 0.75


class QualityDashboard:
    """Real-time quality monitoring dashboard."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """Initialize the quality dashboard."""
        self.thresholds = thresholds or QualityThresholds()
        self.metrics_history: List[QualityMetrics] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.quality_trends: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Initialize trend tracking
        self.quality_trends = {
            'factual_accuracy': [],
            'completeness': [],
            'hallucination': [],
            'utility': [],
            'overall_score': []
        }
    
    def evaluate_response(
        self, 
        response_id: str, 
        query: str, 
        response: str, 
        ground_truth: str,
        source_material: List[str]
    ) -> QualityMetrics:
        """Evaluate a single response and store metrics."""
        try:
            # Run all evaluations
            factual_accuracy = factual_accuracy_evaluator.evaluate(response, ground_truth)
            completeness = completeness_evaluator.evaluate(response, query, ground_truth)
            hallucination = hallucination_detector.detect(response, source_material)
            utility = response_utility_scorer.score(response, query)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                factual_accuracy, completeness, hallucination, utility
            )
            
            # Create metrics object
            metrics = QualityMetrics(
                timestamp=datetime.now(),
                response_id=response_id,
                query=query,
                response=response,
                factual_accuracy=factual_accuracy,
                completeness=completeness,
                hallucination=hallucination,
                utility=utility,
                overall_score=overall_score
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Update trends
            self._update_trends(metrics)
            
            # Check for quality issues
            self._check_quality_issues(metrics)
            
            logger.info(f"Evaluated response {response_id} with overall score {overall_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating response {response_id}: {e}")
            raise
    
    def _calculate_overall_score(
        self,
        factual_accuracy: Dict[str, float],
        completeness: Dict[str, float],
        hallucination: Dict[str, Any],
        utility: Dict[str, float]
    ) -> float:
        """Calculate overall quality score."""
        # Weights for different quality dimensions
        weights = {
            'factual_accuracy': 0.3,
            'completeness': 0.25,
            'hallucination': 0.2,
            'utility': 0.25
        }
        
        # Get individual scores
        factual_score = factual_accuracy.get('overall_score', 0.0)
        completeness_score = completeness.get('overall_completeness', 0.0)
        hallucination_score = 1.0 - hallucination.get('overall_hallucination_score', 0.0)  # Invert hallucination score
        utility_score = utility.get('overall_utility', 0.0)
        
        # Calculate weighted average
        overall_score = (
            factual_score * weights['factual_accuracy'] +
            completeness_score * weights['completeness'] +
            hallucination_score * weights['hallucination'] +
            utility_score * weights['utility']
        )
        
        return round(overall_score, 3)
    
    def _update_trends(self, metrics: QualityMetrics):
        """Update quality trends with new metrics."""
        timestamp = metrics.timestamp
        
        self.quality_trends['factual_accuracy'].append((
            timestamp, 
            metrics.factual_accuracy.get('overall_score', 0.0)
        ))
        self.quality_trends['completeness'].append((
            timestamp, 
            metrics.completeness.get('overall_completeness', 0.0)
        ))
        self.quality_trends['hallucination'].append((
            timestamp, 
            1.0 - metrics.hallucination.get('overall_hallucination_score', 0.0)
        ))
        self.quality_trends['utility'].append((
            timestamp, 
            metrics.utility.get('overall_utility', 0.0)
        ))
        self.quality_trends['overall_score'].append((
            timestamp, 
            metrics.overall_score
        ))
    
    def _check_quality_issues(self, metrics: QualityMetrics):
        """Check for quality issues and generate alerts."""
        issues = []
        
        # Check factual accuracy
        if metrics.factual_accuracy.get('overall_score', 0.0) < self.thresholds.factual_accuracy_min:
            issues.append({
                'type': 'factual_accuracy',
                'severity': 'high',
                'message': f"Factual accuracy {metrics.factual_accuracy.get('overall_score', 0.0):.3f} below threshold {self.thresholds.factual_accuracy_min}",
                'value': metrics.factual_accuracy.get('overall_score', 0.0),
                'threshold': self.thresholds.factual_accuracy_min
            })
        
        # Check completeness
        if metrics.completeness.get('overall_completeness', 0.0) < self.thresholds.completeness_min:
            issues.append({
                'type': 'completeness',
                'severity': 'medium',
                'message': f"Completeness {metrics.completeness.get('overall_completeness', 0.0):.3f} below threshold {self.thresholds.completeness_min}",
                'value': metrics.completeness.get('overall_completeness', 0.0),
                'threshold': self.thresholds.completeness_min
            })
        
        # Check hallucination
        if metrics.hallucination.get('overall_hallucination_score', 0.0) > self.thresholds.hallucination_max:
            issues.append({
                'type': 'hallucination',
                'severity': 'high',
                'message': f"Hallucination score {metrics.hallucination.get('overall_hallucination_score', 0.0):.3f} above threshold {self.thresholds.hallucination_max}",
                'value': metrics.hallucination.get('overall_hallucination_score', 0.0),
                'threshold': self.thresholds.hallucination_max
            })
        
        # Check utility
        if metrics.utility.get('overall_utility', 0.0) < self.thresholds.utility_min:
            issues.append({
                'type': 'utility',
                'severity': 'medium',
                'message': f"Utility score {metrics.utility.get('overall_utility', 0.0):.3f} below threshold {self.thresholds.utility_min}",
                'value': metrics.utility.get('overall_utility', 0.0),
                'threshold': self.thresholds.utility_min
            })
        
        # Check overall score
        if metrics.overall_score < self.thresholds.overall_min:
            issues.append({
                'type': 'overall_score',
                'severity': 'critical',
                'message': f"Overall score {metrics.overall_score:.3f} below threshold {self.thresholds.overall_min}",
                'value': metrics.overall_score,
                'threshold': self.thresholds.overall_min
            })
        
        # Store alerts
        for issue in issues:
            alert = {
                'timestamp': metrics.timestamp,
                'response_id': metrics.response_id,
                'issue': issue
            }
            self.alert_history.append(alert)
            logger.warning(f"Quality issue detected: {issue['message']}")
    
    def get_quality_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get quality summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {
                'total_responses': 0,
                'average_scores': {},
                'quality_issues': 0,
                'trends': {}
            }
        
        # Calculate average scores
        avg_scores = {}
        for metric_name in ['factual_accuracy', 'completeness', 'utility', 'overall_score']:
            if metric_name == 'overall_score':
                values = [m.overall_score for m in recent_metrics]
            else:
                values = []
                for m in recent_metrics:
                    if metric_name == 'factual_accuracy':
                        values.append(m.factual_accuracy.get('overall_score', 0.0))
                    elif metric_name == 'completeness':
                        values.append(m.completeness.get('overall_completeness', 0.0))
                    elif metric_name == 'utility':
                        values.append(m.utility.get('overall_utility', 0.0))
            
            if values:
                avg_scores[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        # Count quality issues
        recent_alerts = [
            a for a in self.alert_history 
            if a['timestamp'] >= cutoff_time
        ]
        
        # Calculate trends
        trends = self._calculate_trends(hours)
        
        return {
            'total_responses': len(recent_metrics),
            'average_scores': avg_scores,
            'quality_issues': len(recent_alerts),
            'trends': trends,
            'thresholds': asdict(self.thresholds)
        }
    
    def _calculate_trends(self, hours: int) -> Dict[str, str]:
        """Calculate quality trends."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        trends = {}
        for metric_name, trend_data in self.quality_trends.items():
            recent_data = [
                (timestamp, value) for timestamp, value in trend_data
                if timestamp >= cutoff_time
            ]
            
            if len(recent_data) < 2:
                trends[metric_name] = 'insufficient_data'
                continue
            
            # Calculate trend direction
            values = [value for _, value in recent_data]
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg * 1.05:
                trends[metric_name] = 'improving'
            elif second_avg < first_avg * 0.95:
                trends[metric_name] = 'declining'
            else:
                trends[metric_name] = 'stable'
        
        return trends
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent quality alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert['timestamp'] >= cutoff_time
        ]
    
    def get_quality_metrics(self, response_id: str) -> Optional[QualityMetrics]:
        """Get quality metrics for a specific response."""
        for metrics in self.metrics_history:
            if metrics.response_id == response_id:
                return metrics
        return None
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export quality metrics to file."""
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump([asdict(m) for m in self.metrics_history], f, indent=2, default=str)
            elif format == 'csv':
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow([
                        'timestamp', 'response_id', 'query', 'response',
                        'factual_accuracy', 'completeness', 'hallucination', 'utility', 'overall_score'
                    ])
                    # Write data
                    for metrics in self.metrics_history:
                        writer.writerow([
                            metrics.timestamp.isoformat(),
                            metrics.response_id,
                            metrics.query,
                            metrics.response,
                            json.dumps(metrics.factual_accuracy),
                            json.dumps(metrics.completeness),
                            json.dumps(metrics.hallucination),
                            json.dumps(metrics.utility),
                            metrics.overall_score
                        ])
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {len(self.metrics_history)} quality metrics to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise
    
    def clear_old_data(self, days: int = 30):
        """Clear old metrics and alerts to save memory."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Clear old metrics
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        # Clear old alerts
        self.alert_history = [
            a for a in self.alert_history
            if a['timestamp'] >= cutoff_time
        ]
        
        # Clear old trend data
        for metric_name in self.quality_trends:
            self.quality_trends[metric_name] = [
                (timestamp, value) for timestamp, value in self.quality_trends[metric_name]
                if timestamp >= cutoff_time
            ]
        
        logger.info(f"Cleared data older than {days} days")


# Global dashboard instance
quality_dashboard = QualityDashboard()
