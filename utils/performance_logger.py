"""
Performance logging utilities for the namer program.
Tracks millisecond-level timing for all operations to identify inefficiencies.
"""

import time
import functools
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class PerformanceLogger:
    """Centralized performance logging for the namer application."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger('namer.performance')
        self.log_file = log_file or 'namer_performance.log'
        self.operation_stack = []
        self.operation_timings = {}
        self.session_start = time.time()
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        # Thread-local storage for operation context
        self._local = threading.local()
        
        # Performance summary
        self.summary = {
            'total_operations': 0,
            'total_time_ms': 0,
            'slow_operations': [],  # Operations > 100ms
            'very_slow_operations': [],  # Operations > 1000ms
            'operation_counts': {},
            'operation_avg_times': {},
            'operation_min_times': {},
            'operation_max_times': {}
        }
    
    def _get_thread_context(self):
        """Get thread-local operation context."""
        if not hasattr(self._local, 'operation_stack'):
            self._local.operation_stack = []
        if not hasattr(self._local, 'operation_timings'):
            self._local.operation_timings = {}
        return self._local.operation_stack, self._local.operation_timings
    
    def start_operation(self, operation_name: str, **kwargs):
        """Start timing an operation."""
        start_time = time.time()
        op_stack, op_timings = self._get_thread_context()
        
        operation_id = f"{operation_name}_{len(op_stack)}"
        op_timings[operation_id] = {
            'name': operation_name,
            'start_time': start_time,
            'kwargs': kwargs,
            'children': []
        }
        op_stack.append(operation_id)
        
        self.logger.info(f"START {operation_name} {kwargs}")
        return operation_id
    
    def end_operation(self, operation_id: str, result: Any = None, error: Optional[Exception] = None):
        """End timing an operation."""
        try:
            end_time = time.time()
            op_stack, op_timings = self._get_thread_context()
            
            if operation_id not in op_timings:
                self.logger.warning(f"Operation {operation_id} not found in timings")
                return None
            
            timing_data = op_timings[operation_id]
            duration_ms = (end_time - timing_data['start_time']) * 1000
            
            timing_data.update({
                'end_time': end_time,
                'duration_ms': duration_ms,
                'result': result,
                'error': str(error) if error else None
            })
            
            # Remove from stack
            if op_stack and op_stack[-1] == operation_id:
                op_stack.pop()
            
            # Update summary
            self.summary['total_operations'] += 1
            self.summary['total_time_ms'] += duration_ms
            
            op_name = timing_data['name']
            if op_name not in self.summary['operation_counts']:
                self.summary['operation_counts'][op_name] = 0
                self.summary['operation_avg_times'][op_name] = 0
                self.summary['operation_min_times'][op_name] = float('inf')
                self.summary['operation_max_times'][op_name] = 0
            
            self.summary['operation_counts'][op_name] += 1
            count = self.summary['operation_counts'][op_name]
            
            # Update running averages
            current_avg = self.summary['operation_avg_times'][op_name]
            self.summary['operation_avg_times'][op_name] = (current_avg * (count - 1) + duration_ms) / count
            
            # Update min/max
            self.summary['operation_min_times'][op_name] = min(self.summary['operation_min_times'][op_name], duration_ms)
            self.summary['operation_max_times'][op_name] = max(self.summary['operation_max_times'][op_name], duration_ms)
            
            # Track slow operations
            if duration_ms > 1000:
                self.summary['very_slow_operations'].append({
                    'operation': op_name,
                    'duration_ms': duration_ms,
                    'kwargs': timing_data['kwargs'],
                    'error': timing_data['error']
                })
            elif duration_ms > 100:
                self.summary['slow_operations'].append({
                    'operation': op_name,
                    'duration_ms': duration_ms,
                    'kwargs': timing_data['kwargs']
                })
            
            # Log with appropriate level based on duration
            if duration_ms > 1000:
                self.logger.warning(f"END {op_name} {duration_ms:.1f}ms (VERY SLOW) {timing_data['kwargs']}")
            elif duration_ms > 100:
                self.logger.info(f"END {op_name} {duration_ms:.1f}ms (SLOW) {timing_data['kwargs']}")
            else:
                self.logger.debug(f"END {op_name} {duration_ms:.1f}ms {timing_data['kwargs']}")
            
            return timing_data
            
        except Exception as e:
            self.logger.error(f"Error in end_operation for {operation_id}: {e}")
            return None
    
    @contextmanager
    def operation(self, operation_name: str, **kwargs):
        """Context manager for timing operations."""
        op_id = self.start_operation(operation_name, **kwargs)
        try:
            yield op_id
        except Exception as e:
            self.end_operation(op_id, error=e)
            raise
        else:
            self.end_operation(op_id)
    
    def time_function(self, operation_name: Optional[str] = None):
        """Decorator to time function execution."""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                with self.operation(name, args_count=len(args), kwargs_keys=list(kwargs.keys())):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        session_duration = (time.time() - self.session_start) * 1000
        
        summary = self.summary.copy()
        summary.update({
            'session_id': self.session_id,
            'session_duration_ms': session_duration,
            'operations_per_second': self.summary['total_operations'] / (session_duration / 1000) if session_duration > 0 else 0,
            'avg_operation_time_ms': self.summary['total_time_ms'] / self.summary['total_operations'] if self.summary['total_operations'] > 0 else 0
        })
        
        return summary
    
    def save_summary(self, filename: Optional[str] = None):
        """Save performance summary to file."""
        if filename is None:
            filename = f"namer_performance_summary_{self.session_id}.json"
        
        summary = self.get_summary()
        summary['timestamp'] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Performance summary saved to {filename}")
        return filename
    
    def print_summary(self):
        """Print performance summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("NAMER PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Session ID: {summary['session_id']}")
        print(f"Session Duration: {summary['session_duration_ms']:.1f}ms")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Time: {summary['total_time_ms']:.1f}ms")
        print(f"Avg Operation Time: {summary['avg_operation_time_ms']:.1f}ms")
        print(f"Operations/Second: {summary['operations_per_second']:.2f}")
        
        if summary['slow_operations']:
            print(f"\nSLOW OPERATIONS (>100ms): {len(summary['slow_operations'])}")
            for op in sorted(summary['slow_operations'], key=lambda x: x['duration_ms'], reverse=True)[:10]:
                print(f"  {op['operation']}: {op['duration_ms']:.1f}ms")
        
        if summary['very_slow_operations']:
            print(f"\nVERY SLOW OPERATIONS (>1000ms): {len(summary['very_slow_operations'])}")
            for op in sorted(summary['very_slow_operations'], key=lambda x: x['duration_ms'], reverse=True)[:5]:
                print(f"  {op['operation']}: {op['duration_ms']:.1f}ms")
        
        print(f"\nOPERATION BREAKDOWN:")
        for op_name, count in sorted(summary['operation_counts'].items(), key=lambda x: x[1], reverse=True):
            avg_time = summary['operation_avg_times'][op_name]
            min_time = summary['operation_min_times'][op_name]
            max_time = summary['operation_max_times'][op_name]
            print(f"  {op_name}: {count} calls, avg: {avg_time:.1f}ms, min: {min_time:.1f}ms, max: {max_time:.1f}ms")
        
        print("="*80)

# Global performance logger instance
performance_logger = PerformanceLogger()

# Convenience functions
def time_operation(operation_name: str, **kwargs):
    """Convenience function to time an operation."""
    return performance_logger.operation(operation_name, **kwargs)

def time_function(operation_name: Optional[str] = None):
    """Convenience decorator to time a function."""
    return performance_logger.time_function(operation_name)

def get_performance_summary():
    """Get current performance summary."""
    return performance_logger.get_summary()

def print_performance_summary():
    """Print current performance summary."""
    performance_logger.print_summary()

def save_performance_summary(filename: Optional[str] = None):
    """Save performance summary to file."""
    return performance_logger.save_summary(filename)

def reset_performance_data():
    """Reset all performance data."""
    global performance_logger
    performance_logger = PerformanceLogger() 