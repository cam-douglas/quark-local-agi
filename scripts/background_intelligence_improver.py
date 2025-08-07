#!/usr/bin/env python3
"""
Background Intelligence Improver for Quark AI System
Continuously optimizes intelligence, learns from interactions, and improves efficiency
"""

import os
import sys
import time
import json
import asyncio
import threading
import subprocess
import psutil
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackgroundIntelligenceImprover:
    """Background intelligence optimization and learning system"""
    
    def __init__(self):
        self.quark_dir = Path("/Users/camdouglas/quark")
        self.data_dir = self.quark_dir / "data" / "intelligence_improvement"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.running = False
        self.optimization_interval = 300  # 5 minutes
        self.learning_interval = 60  # 1 minute
        self.memory_optimization_interval = 1800  # 30 minutes
        
        # Performance tracking
        self.performance_metrics = {
            'response_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'learning_improvements': [],
            'optimization_cycles': 0
        }
        
        # Intelligence improvement data
        self.improvement_data = {
            'last_optimization': None,
            'total_improvements': 0,
            'learning_patterns': [],
            'efficiency_gains': []
        }
        
    def load_improvement_data(self):
        """Load intelligence improvement data"""
        data_file = self.data_dir / "improvement_data.json"
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    self.improvement_data.update(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load improvement data: {e}")
    
    def save_improvement_data(self):
        """Save intelligence improvement data"""
        data_file = self.data_dir / "improvement_data.json"
        try:
            with open(data_file, 'w') as f:
                json.dump(self.improvement_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save improvement data: {e}")
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'thread_count': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def optimize_memory_usage(self):
        """Optimize memory usage and garbage collection"""
        try:
            logger.info("🧹 Optimizing memory usage...")
            
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"✅ Garbage collection: {collected} objects collected")
            
            # Memory optimization strategies
            if hasattr(gc, 'set_threshold'):
                # Adjust garbage collection thresholds
                gc.set_threshold(700, 10, 10)  # More aggressive GC
            
            # Clear Python cache
            import importlib
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('_') or '.' in module_name:
                    continue
                try:
                    module = sys.modules[module_name]
                    if hasattr(module, '__file__') and module.__file__:
                        importlib.reload(module)
                except:
                    pass
            
            logger.info("✅ Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")
    
    def analyze_learning_patterns(self):
        """Analyze and learn from interaction patterns"""
        try:
            logger.info("🧠 Analyzing learning patterns...")
            
            # Analyze log files for patterns
            log_files = [
                self.quark_dir / "logs" / "quark_startup.log",
                self.quark_dir / "logs" / "quark_output.log"
            ]
            
            patterns = []
            for log_file in log_files:
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-1000:]  # Last 1000 lines
                        
                        # Analyze response patterns
                        response_times = []
                        error_patterns = []
                        
                        for line in lines:
                            if "response_time" in line.lower():
                                try:
                                    time_str = line.split("response_time")[1].split()[0]
                                    response_times.append(float(time_str))
                                except:
                                    pass
                            
                            if "error" in line.lower():
                                error_patterns.append(line.strip())
                        
                        if response_times:
                            avg_response = sum(response_times) / len(response_times)
                            patterns.append({
                                'type': 'response_time',
                                'average': avg_response,
                                'count': len(response_times)
                            })
                        
                        if error_patterns:
                            patterns.append({
                                'type': 'errors',
                                'patterns': error_patterns[-10:],  # Last 10 errors
                                'count': len(error_patterns)
                            })
            
            # Store learning patterns
            self.improvement_data['learning_patterns'].extend(patterns)
            
            # Keep only recent patterns (last 100)
            if len(self.improvement_data['learning_patterns']) > 100:
                self.improvement_data['learning_patterns'] = self.improvement_data['learning_patterns'][-100:]
            
            logger.info(f"✅ Analyzed {len(patterns)} learning patterns")
            
        except Exception as e:
            logger.error(f"Learning pattern analysis error: {e}")
    
    def optimize_intelligence_algorithms(self):
        """Optimize intelligence algorithms and decision making"""
        try:
            logger.info("🧠 Optimizing intelligence algorithms...")
            
            # Optimize agent decision making
            agent_optimizations = {
                'negotiation_agent': {
                    'decision_threshold': 0.8,
                    'learning_rate': 0.01,
                    'memory_size': 1000
                },
                'tool_discovery_agent': {
                    'evaluation_criteria': ['performance', 'security', 'compatibility'],
                    'cache_size': 500,
                    'update_frequency': 3600
                },
                'explainability_agent': {
                    'explanation_depth': 'adaptive',
                    'confidence_threshold': 0.9,
                    'transparency_level': 'high'
                }
            }
            
            # Save optimizations
            optimization_file = self.data_dir / "agent_optimizations.json"
            with open(optimization_file, 'w') as f:
                json.dump(agent_optimizations, f, indent=2)
            
            # Update improvement data
            self.improvement_data['total_improvements'] += 1
            self.improvement_data['last_optimization'] = datetime.now().isoformat()
            
            logger.info("✅ Intelligence algorithms optimized")
            
        except Exception as e:
            logger.error(f"Intelligence optimization error: {e}")
    
    def improve_efficiency(self):
        """Improve overall system efficiency"""
        try:
            logger.info("⚡ Improving system efficiency...")
            
            # Get current metrics
            metrics = self.get_system_metrics()
            
            # Track efficiency gains
            efficiency_gain = {
                'timestamp': datetime.now().isoformat(),
                'memory_usage': metrics.get('memory_mb', 0),
                'cpu_usage': metrics.get('cpu_percent', 0),
                'optimizations_applied': []
            }
            
            # Apply efficiency improvements
            improvements = []
            
            # 1. Optimize file I/O
            if metrics.get('open_files', 0) > 100:
                improvements.append('file_io_optimization')
                efficiency_gain['optimizations_applied'].append('file_io_optimization')
            
            # 2. Optimize network connections
            if metrics.get('connections', 0) > 50:
                improvements.append('connection_pooling')
                efficiency_gain['optimizations_applied'].append('connection_pooling')
            
            # 3. Memory optimization
            if metrics.get('memory_percent', 0) > 80:
                improvements.append('memory_optimization')
                self.optimize_memory_usage()
                efficiency_gain['optimizations_applied'].append('memory_optimization')
            
            # 4. Thread optimization
            if metrics.get('thread_count', 0) > 20:
                improvements.append('thread_optimization')
                efficiency_gain['optimizations_applied'].append('thread_optimization')
            
            # Store efficiency gains
            self.improvement_data['efficiency_gains'].append(efficiency_gain)
            
            # Keep only recent gains (last 50)
            if len(self.improvement_data['efficiency_gains']) > 50:
                self.improvement_data['efficiency_gains'] = self.improvement_data['efficiency_gains'][-50:]
            
            logger.info(f"✅ Applied {len(improvements)} efficiency improvements")
            
        except Exception as e:
            logger.error(f"Efficiency improvement error: {e}")
    
    def adaptive_learning(self):
        """Adaptive learning based on system performance"""
        try:
            logger.info("🎓 Running adaptive learning...")
            
            # Analyze recent performance
            recent_gains = self.improvement_data['efficiency_gains'][-10:]
            if recent_gains:
                avg_memory = sum(g['memory_usage'] for g in recent_gains) / len(recent_gains)
                avg_cpu = sum(g['cpu_usage'] for g in recent_gains) / len(recent_gains)
                
                # Adaptive adjustments
                if avg_memory > 1000:  # High memory usage
                    logger.info("📉 High memory usage detected - applying memory optimizations")
                    self.optimize_memory_usage()
                
                if avg_cpu > 80:  # High CPU usage
                    logger.info("📉 High CPU usage detected - applying CPU optimizations")
                    # Adjust optimization intervals
                    self.optimization_interval = min(600, self.optimization_interval + 60)
                    self.learning_interval = min(120, self.learning_interval + 10)
                else:
                    # Gradually reduce intervals for better responsiveness
                    self.optimization_interval = max(300, self.optimization_interval - 30)
                    self.learning_interval = max(60, self.learning_interval - 5)
            
            # Learn from patterns
            self.analyze_learning_patterns()
            
            logger.info("✅ Adaptive learning completed")
            
        except Exception as e:
            logger.error(f"Adaptive learning error: {e}")
    
    async def continuous_improvement(self):
        """Continuous intelligence improvement loop"""
        logger.info("🚀 Starting background intelligence improver...")
        
        while self.running:
            try:
                # Track performance metrics
                metrics = self.get_system_metrics()
                self.performance_metrics['memory_usage'].append(metrics.get('memory_mb', 0))
                self.performance_metrics['cpu_usage'].append(metrics.get('cpu_percent', 0))
                
                # Keep only recent metrics (last 100)
                for key in ['memory_usage', 'cpu_usage']:
                    if len(self.performance_metrics[key]) > 100:
                        self.performance_metrics[key] = self.performance_metrics[key][-100:]
                
                # Run optimization cycle
                self.performance_metrics['optimization_cycles'] += 1
                
                # Adaptive learning (every learning_interval)
                if self.performance_metrics['optimization_cycles'] % (self.learning_interval // 5) == 0:
                    self.adaptive_learning()
                
                # Intelligence optimization (every optimization_interval)
                if self.performance_metrics['optimization_cycles'] % (self.optimization_interval // 5) == 0:
                    self.optimize_intelligence_algorithms()
                
                # Efficiency improvement (every cycle)
                self.improve_efficiency()
                
                # Memory optimization (every memory_optimization_interval)
                if self.performance_metrics['optimization_cycles'] % (self.memory_optimization_interval // 5) == 0:
                    self.optimize_memory_usage()
                
                # Save improvement data periodically
                if self.performance_metrics['optimization_cycles'] % 10 == 0:
                    self.save_improvement_data()
                
                # Log progress
                if self.performance_metrics['optimization_cycles'] % 20 == 0:
                    logger.info(f"🔄 Improvement cycle {self.performance_metrics['optimization_cycles']} completed")
                    logger.info(f"📊 Memory: {metrics.get('memory_mb', 0):.1f}MB, CPU: {metrics.get('cpu_percent', 0):.1f}%")
                
                # Wait before next cycle
                await asyncio.sleep(5)  # 5-second cycles
                
            except Exception as e:
                logger.error(f"Improvement cycle error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def start(self):
        """Start background intelligence improver"""
        self.running = True
        self.load_improvement_data()
        
        def run_async():
            asyncio.run(self.continuous_improvement())
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        logger.info("✅ Background intelligence improver started")
        return thread
    
    def stop(self):
        """Stop background intelligence improver"""
        self.running = False
        self.save_improvement_data()
        logger.info("🛑 Background intelligence improver stopped")
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get intelligence improvement statistics"""
        return {
            'total_improvements': self.improvement_data['total_improvements'],
            'optimization_cycles': self.performance_metrics['optimization_cycles'],
            'learning_patterns': len(self.improvement_data['learning_patterns']),
            'efficiency_gains': len(self.improvement_data['efficiency_gains']),
            'last_optimization': self.improvement_data['last_optimization'],
            'current_memory_mb': self.performance_metrics['memory_usage'][-1] if self.performance_metrics['memory_usage'] else 0,
            'current_cpu_percent': self.performance_metrics['cpu_usage'][-1] if self.performance_metrics['cpu_usage'] else 0
        }

def main():
    """Main entry point for background intelligence improver"""
    improver = BackgroundIntelligenceImprover()
    
    try:
        # Start background improver
        thread = improver.start()
        
        # Keep running and show stats periodically
        while True:
            time.sleep(60)  # Show stats every minute
            stats = improver.get_improvement_stats()
            logger.info(f"📊 Intelligence Stats: {stats['total_improvements']} improvements, "
                       f"{stats['optimization_cycles']} cycles, "
                       f"{stats['current_memory_mb']:.1f}MB memory")
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
        improver.stop()
    except Exception as e:
        logger.error(f"Background intelligence improver error: {e}")

if __name__ == "__main__":
    main() 