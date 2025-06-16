import numpy as np
import wandb
from datetime import datetime
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from config import Config

@dataclass
class QueueUpdateEvent:
    """Represents a single queue update event"""
    timestamp: float
    event_type: str  # 'request_added', 'request_completed', 'request_failed'
    server_id: int
    request_id: str
    prompt_preview: str  # First 100 chars of prompt
    queue_length_before: int = 0
    queue_length_after: int = 0
    utilization_before: float = 0
    utilization_after: float = 0
    processing_latency: Optional[float] = None
    quality_score: Optional[float] = None
    reward: Optional[float] = None
    episode: Optional[int] = None

@dataclass
class ServerQueueState:
    """Current state of a server's queue"""
    server_id: int
    current_load: int
    capacity: int
    utilization: float
    pending_completions: int
    avg_processing_time: float
    recent_requests: int
    active_request_ids: List[str]

class QueueUpdateMonitor:
    """Monitor and log every queue update event"""
    
    def __init__(self, wandb_available=False, max_events=1000):
        self.wandb_available = wandb_available
        self.max_events = max_events
        
        # Event storage
        self.queue_events = deque(maxlen=max_events)
        self.server_states = {}
        
        # Statistics tracking
        self.event_counts = {
            'request_added': 0,
            'request_completed': 0,
            'request_failed': 0
        }
        
        # Real-time metrics
        self.current_episode = 0  # Initialize episode tracking
        self.last_log_time = 0
        self.log_interval = 5.0  # Log summary every 5 seconds
        
        # Initialize server states
        for i in range(len(Config.SERVER_CAPACITIES)):
            self.server_states[i] = ServerQueueState(
                server_id=i,
                current_load=0,
                capacity=Config.SERVER_CAPACITIES[i],
                utilization=0.0,
                pending_completions=0,
                avg_processing_time=0.0,
                recent_requests=0,
                active_request_ids=[]
            )
    
    def log_request_added(self, server_id: int, request_id: str, prompt: str, 
                         current_time: float, processing_latency: float, 
                         quality_score: float, queue_state_before: Dict, 
                         queue_state_after: Dict, episode: int = None):
        """Log when a request is added to a server queue"""
        
        event = QueueUpdateEvent(
            timestamp=current_time,
            event_type='request_added',
            server_id=server_id,
            request_id=request_id,
            prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt,
            queue_length_before=queue_state_before['current_load'],
            queue_length_after=queue_state_after['current_load'],
            utilization_before=queue_state_before['utilization'],
            utilization_after=queue_state_after['utilization'],
            processing_latency=processing_latency,
            quality_score=quality_score,
            episode=episode
        )
        
        self._add_event(event)
        self._update_server_state(server_id, queue_state_after)
        
        # Log to wandb immediately for real-time monitoring
        if self.wandb_available:
            self._log_event_to_wandb(event, "ADDED")
    
    def log_request_completed(self, server_id: int, request_id: str, 
                            current_time: float, reward: float,
                            queue_state_before: Dict, queue_state_after: Dict,
                            episode: int = None):
        """Log when a request is completed by a server"""
        
        event = QueueUpdateEvent(
            timestamp=current_time,
            event_type='request_completed',
            server_id=server_id,
            request_id=request_id,
            prompt_preview="<completed>",
            queue_length_before=queue_state_before['current_load'],
            queue_length_after=queue_state_after['current_load'],
            utilization_before=queue_state_before['utilization'],
            utilization_after=queue_state_after['utilization'],
            reward=reward,
            episode=episode
        )
        
        self._add_event(event)
        self._update_server_state(server_id, queue_state_after)
        
        # Log to wandb immediately
        if self.wandb_available:
            self._log_event_to_wandb(event, "COMPLETED")
    
    def log_request_failed(self, server_id: int, request_id: str, prompt: str,
                          current_time: float, reason: str, episode: int = None):
        """Log when a request fails (e.g., server at capacity)"""
        
        event = QueueUpdateEvent(
            timestamp=current_time,
            event_type='request_failed',
            server_id=server_id,
            request_id=request_id,
            prompt_preview=f"[FAILED: {reason}] {prompt[:80]}...",
            queue_length_before=-1,  # Not applicable
            queue_length_after=-1,   # Not applicable
            utilization_before=-1,   # Not applicable
            utilization_after=-1,    # Not applicable
            episode=episode
        )
        
        self._add_event(event)
        
        # Log to wandb immediately
        if self.wandb_available:
            self._log_event_to_wandb(event, "FAILED")
    
    def _add_event(self, event: QueueUpdateEvent):
        """Add event to storage and update statistics"""
        self.queue_events.append(event)
        self.event_counts[event.event_type] += 1
        
        # Print real-time update to console
        self._print_event_update(event)
    
    def _update_server_state(self, server_id: int, queue_info: Dict):
        """Update cached server state"""
        state = self.server_states[server_id]
        state.current_load = queue_info['current_load']
        state.utilization = queue_info['utilization']
        state.pending_completions = queue_info['pending_completions']
        state.avg_processing_time = queue_info['avg_processing_time']
        state.recent_requests = queue_info.get('recent_requests_per_minute', 0)
    
    def _print_event_update(self, event: QueueUpdateEvent):
        """Print real-time event update to console"""
        # Check if console queue events are enabled
        try:
            if not Config.CONSOLE_CONFIG.get('queue_events', True):
                return  # Don't print if disabled
        except:
            pass  # If config not available, print anyway
        
        timestamp_str = f"{event.timestamp:.2f}s"
        
        if event.event_type == 'request_added':
            print(f"[{timestamp_str}] ➕ ADDED   | Server {event.server_id} | "
                  f"Queue: {event.queue_length_before}→{event.queue_length_after} | "
                  f"Util: {event.utilization_before:.2f}→{event.utilization_after:.2f} | "
                  f"Request: {event.request_id} | Quality: {event.quality_score:.3f}")
        
        elif event.event_type == 'request_completed':
            print(f"[{timestamp_str}] ✅ COMPLETED | Server {event.server_id} | "
                  f"Queue: {event.queue_length_before}→{event.queue_length_after} | "
                  f"Util: {event.utilization_before:.2f}→{event.utilization_after:.2f} | "
                  f"Request: {event.request_id} | Reward: {event.reward:.3f}")
        
        elif event.event_type == 'request_failed':
            print(f"[{timestamp_str}] ❌ FAILED  | Server {event.server_id} | "
                  f"Request: {event.request_id} | Reason: {event.prompt_preview}")
    
    def _log_event_to_wandb(self, event: QueueUpdateEvent, action_type: str):
        """Log individual event to wandb for real-time monitoring"""
        try:
            log_dict = {
                f'queue_event_timestamp': event.timestamp,
                f'queue_event_type': event.event_type,
                f'queue_event_server': event.server_id,
                f'queue_event_action': action_type,
            }
            
            # Add specific metrics based on event type
            if event.event_type in ['request_added', 'request_completed']:
                log_dict.update({
                    f'queue_length_change_server_{event.server_id}': 
                        event.queue_length_after - event.queue_length_before,
                    f'utilization_change_server_{event.server_id}': 
                        event.utilization_after - event.utilization_before,
                    f'current_queue_length_server_{event.server_id}': event.queue_length_after,
                    f'current_utilization_server_{event.server_id}': event.utilization_after,
                })
            
            if event.quality_score is not None:
                log_dict[f'event_quality_score'] = event.quality_score
            
            if event.reward is not None:
                log_dict[f'event_reward'] = event.reward
            
            if event.episode is not None:
                log_dict['episode'] = event.episode
            
            # Add current system state
            total_queue_length = sum(state.current_load for state in self.server_states.values())
            avg_utilization = np.mean([state.utilization for state in self.server_states.values()])
            
            log_dict.update({
                'queue_system_total_length_realtime': total_queue_length,
                'queue_system_avg_utilization_realtime': avg_utilization,
                'queue_events_total': len(self.queue_events),
                'queue_events_added_total': self.event_counts['request_added'],
                'queue_events_completed_total': self.event_counts['request_completed'],
                'queue_events_failed_total': self.event_counts['request_failed'],
            })
            
            wandb.log(log_dict)
            
        except Exception as e:
            print(f"Failed to log queue event to wandb: {e}")
    
    def log_periodic_summary(self, current_time: float, episode: int):
        """Log periodic summary of queue activity"""
        if current_time - self.last_log_time < self.log_interval:
            return
        
        # Check if console periodic summaries are enabled
        try:
            console_enabled = Config.CONSOLE_CONFIG.get('periodic_summaries', True)
        except:
            console_enabled = True  # Default to enabled if config not available
        
        if console_enabled:
            print(f"\n📊 QUEUE SUMMARY at {current_time:.2f}s (Episode {episode})")
            print("=" * 60)
            
            # Server states
            for server_id, state in self.server_states.items():
                print(f"Server {server_id}: "
                      f"Queue={state.current_load}/{state.capacity} "
                      f"({state.utilization:.1%}) | "
                      f"Pending={state.pending_completions} | "
                      f"AvgTime={state.avg_processing_time:.2f}s")
            
            # Event statistics
            total_events = sum(self.event_counts.values())
            if total_events > 0:
                print(f"\nEvent Counts:")
                print(f"  Added: {self.event_counts['request_added']}")
                print(f"  Completed: {self.event_counts['request_completed']}")
                print(f"  Failed: {self.event_counts['request_failed']}")
                
                # Calculate rates
                success_rate = self.event_counts['request_completed'] / max(1, self.event_counts['request_added'])
                failure_rate = self.event_counts['request_failed'] / max(1, total_events)
                
                print(f"  Success Rate: {success_rate:.1%}")
                print(f"  Failure Rate: {failure_rate:.1%}")
            
            print("=" * 60)
        
        self.last_log_time = current_time
        
        # Log summary to wandb (always do this regardless of console setting)
        if self.wandb_available:
            self._log_summary_to_wandb(current_time, episode)
    
    def _log_summary_to_wandb(self, current_time: float, episode: int):
        """Log periodic summary to wandb"""
        try:
            summary_dict = {
                'queue_summary_timestamp': current_time,
                'episode': episode,
            }
            
            # Server utilizations
            for server_id, state in self.server_states.items():
                summary_dict.update({
                    f'queue_summary_server_{server_id}_load': state.current_load,
                    f'queue_summary_server_{server_id}_utilization': state.utilization,
                    f'queue_summary_server_{server_id}_pending': state.pending_completions,
                })
            
            # Event statistics
            total_events = sum(self.event_counts.values())
            if total_events > 0:
                success_rate = self.event_counts['request_completed'] / max(1, self.event_counts['request_added'])
                failure_rate = self.event_counts['request_failed'] / max(1, total_events)
                
                summary_dict.update({
                    'queue_summary_success_rate': success_rate,
                    'queue_summary_failure_rate': failure_rate,
                    'queue_summary_total_events': total_events,
                })
            
            wandb.log(summary_dict)
            
        except Exception as e:
            print(f"Failed to log queue summary to wandb: {e}")
    
    def log_queue_trends(self, episode: int):
        """Log queue trends and patterns over recent history"""
        if not self.wandb_available:
            return
        
        try:
            trend_dict = {'episode': episode}
            
            # Calculate trends for each server
            for server_id in range(len(Config.SERVER_CAPACITIES)):
                server_state = self.server_states.get(server_id)
                if server_state:
                    # Recent events for this server
                    recent_events = [event for event in list(self.queue_events)[-50:] 
                                   if event.server_id == server_id]
                    
                    if len(recent_events) >= 5:
                        # Queue length trend (recent vs earlier)
                        added_events = [e for e in recent_events if e.event_type == 'request_added']
                        completed_events = [e for e in recent_events if e.event_type == 'request_completed']
                        
                        if len(added_events) >= 3:
                            recent_queue_changes = [e.queue_length_after - e.queue_length_before for e in added_events[-3:]]
                            earlier_queue_changes = [e.queue_length_after - e.queue_length_before for e in added_events[-6:-3]] if len(added_events) >= 6 else recent_queue_changes
                            
                            queue_trend = np.mean(recent_queue_changes) - np.mean(earlier_queue_changes)
                        else:
                            queue_trend = 0.0
                        
                        # Utilization trend
                        if len(added_events) >= 2:
                            recent_util = np.mean([e.utilization_after for e in added_events[-2:]])
                            earlier_util = np.mean([e.utilization_after for e in added_events[-4:-2]]) if len(added_events) >= 4 else recent_util
                            util_trend = recent_util - earlier_util
                        else:
                            util_trend = 0.0
                        
                        # Average processing metrics
                        avg_quality = np.mean([e.quality_score for e in added_events if e.quality_score]) if added_events else 0.0
                        avg_reward = np.mean([e.reward for e in completed_events if e.reward]) if completed_events else 0.0
                        
                        trend_dict.update({
                            f'server_{server_id}_queue_trend': queue_trend,
                            f'server_{server_id}_utilization_trend': util_trend,
                            f'server_{server_id}_recent_avg_quality': avg_quality,
                            f'server_{server_id}_recent_avg_reward': avg_reward,
                            f'server_{server_id}_recent_events_count': len(recent_events),
                            f'server_{server_id}_current_utilization': server_state.utilization,
                            f'server_{server_id}_current_load': server_state.current_load,
                        })
            
            # System-wide trends
            if len(self.queue_events) >= 10:
                recent_system_events = list(self.queue_events)[-20:]
                
                # Event rate trends
                added_count = sum(1 for e in recent_system_events if e.event_type == 'request_added')
                completed_count = sum(1 for e in recent_system_events if e.event_type == 'request_completed')
                failed_count = sum(1 for e in recent_system_events if e.event_type == 'request_failed')
                
                trend_dict.update({
                    'system_recent_added_rate': added_count / 20,
                    'system_recent_completed_rate': completed_count / 20,
                    'system_recent_failed_rate': failed_count / 20,
                    'system_recent_success_rate': completed_count / max(1, added_count),
                })
            
            wandb.log(trend_dict)
            
        except Exception as e:
            print(f"Failed to log queue trends to wandb: {e}")
    
    def get_recent_events(self, num_events: int = 10) -> List[QueueUpdateEvent]:
        """Get the most recent queue events"""
        return list(self.queue_events)[-num_events:]
    
    def get_server_activity(self, server_id: int, time_window: float = 60.0) -> Dict:
        """Get recent activity for a specific server"""
        current_time = self.queue_events[-1].timestamp if self.queue_events else 0
        cutoff_time = current_time - time_window
        
        recent_events = [event for event in self.queue_events 
                        if event.server_id == server_id and event.timestamp >= cutoff_time]
        
        added_events = [e for e in recent_events if e.event_type == 'request_added']
        completed_events = [e for e in recent_events if e.event_type == 'request_completed']
        failed_events = [e for e in recent_events if e.event_type == 'request_failed']
        
        return {
            'server_id': server_id,
            'time_window': time_window,
            'requests_added': len(added_events),
            'requests_completed': len(completed_events),
            'requests_failed': len(failed_events),
            'avg_quality_score': np.mean([e.quality_score for e in added_events if e.quality_score]),
            'avg_reward': np.mean([e.reward for e in completed_events if e.reward]),
            'current_state': self.server_states[server_id]
        }
    
    def export_events_to_file(self, filename: str = None):
        """Export all events to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"queue_events_{timestamp}.json"
        
        import json
        
        events_data = [asdict(event) for event in self.queue_events]
        
        with open(filename, 'w') as f:
            json.dump({
                'events': events_data,
                'statistics': self.event_counts,
                'server_states': {k: asdict(v) for k, v in self.server_states.items()}
            }, f, indent=2)
        
        print(f"Exported {len(events_data)} queue events to {filename}")
        
        return filename
    
    def create_queue_visualization(self):
        """Create visualization of queue states over time"""
        if not self.queue_events:
            print("No queue data available for visualization")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Prepare data by server
            timestamps = {}
            queue_lengths = {}
            utilizations = {}
            
            for server_id in range(len(Config.SERVER_CAPACITIES)):
                server_events = [e for e in self.queue_events if e.server_id == server_id]
                server_events.sort(key=lambda x: x.timestamp)
                
                timestamps[server_id] = [e.timestamp for e in server_events if e.event_type in ['request_added', 'request_completed']]
                queue_lengths[server_id] = [e.queue_length_after for e in server_events if e.event_type in ['request_added', 'request_completed'] and e.queue_length_after >= 0]
                utilizations[server_id] = [e.utilization_after for e in server_events if e.event_type in ['request_added', 'request_completed'] and e.utilization_after >= 0]
            
            # Plot 1: Queue lengths over time
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            for server_id in range(len(Config.SERVER_CAPACITIES)):
                if timestamps[server_id] and queue_lengths[server_id]:
                    axes[0, 0].plot(timestamps[server_id][:len(queue_lengths[server_id])], 
                                  queue_lengths[server_id], 
                                  label=f'Server {server_id}', 
                                  linewidth=2, 
                                  color=colors[server_id % len(colors)])
                    # Add capacity line
                    capacity = Config.SERVER_CAPACITIES[server_id]
                    axes[0, 0].axhline(y=capacity, color=colors[server_id % len(colors)], 
                                     linestyle='--', alpha=0.5)
            
            axes[0, 0].set_title('Queue Lengths Over Time')
            axes[0, 0].set_xlabel('Simulation Time')
            axes[0, 0].set_ylabel('Queue Length')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Utilization over time
            for server_id in range(len(Config.SERVER_CAPACITIES)):
                if timestamps[server_id] and utilizations[server_id]:
                    axes[0, 1].plot(timestamps[server_id][:len(utilizations[server_id])], 
                                  utilizations[server_id], 
                                  label=f'Server {server_id}', 
                                  linewidth=2,
                                  color=colors[server_id % len(colors)])
            
            axes[0, 1].set_title('Server Utilization Over Time')
            axes[0, 1].set_xlabel('Simulation Time')
            axes[0, 1].set_ylabel('Utilization')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1.1)
            
            # Plot 3: Event type distribution
            event_types = ['request_added', 'request_completed', 'request_failed']
            event_counts = [self.event_counts[et] for et in event_types]
            
            if sum(event_counts) > 0:
                axes[1, 0].bar(event_types, event_counts, 
                             color=['#4CAF50', '#2196F3', '#F44336'], alpha=0.7)
                axes[1, 0].set_title('Event Type Distribution')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Server activity summary
            server_ids = list(range(len(Config.SERVER_CAPACITIES)))
            server_loads = [self.server_states[i].current_load for i in server_ids]
            server_utils = [self.server_states[i].utilization for i in server_ids]
            
            x = np.arange(len(server_ids))
            width = 0.35
            
            bars1 = axes[1, 1].bar(x - width/2, server_loads, width, 
                                 label='Current Load', alpha=0.8, color='#FF6B6B')
            
            # Secondary y-axis for utilization
            ax2 = axes[1, 1].twinx()
            bars2 = ax2.bar(x + width/2, server_utils, width, 
                          label='Utilization', alpha=0.8, color='#4ECDC4')
            
            axes[1, 1].set_title('Current Server Status')
            axes[1, 1].set_xlabel('Server ID')
            axes[1, 1].set_ylabel('Queue Load')
            ax2.set_ylabel('Utilization')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels([f'S{i}' for i in server_ids])
            
            # Combined legend
            lines1, labels1 = axes[1, 1].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            plt.savefig('queue_monitoring.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print("Queue monitoring visualization saved to queue_monitoring.png")
            
            # Log to wandb if available
            if self.wandb_available:
                try:
                    wandb.log({"queue_monitoring": wandb.Image('queue_monitoring.png')})
                    print("Queue monitoring plot uploaded to wandb")
                except Exception as e:
                    print(f"Failed to upload queue monitoring to wandb: {e}")
                    
        except ImportError:
            print("Matplotlib not available - cannot create queue visualization")
        except Exception as e:
            print(f"Queue visualization failed: {e}")
    
    def print_queue_summary(self):
        """Print a human-readable queue summary"""
        summary = self.get_queue_summary()
        
        print("\nQueue State Summary:")
        print("=" * 60)
        
        # Server states
        for server_id, state in self.server_states.items():
            capacity = Config.SERVER_CAPACITIES[server_id]
            
            print(f"\nServer {server_id} (Capacity: {capacity}):")
            print(f"  Current Load: {state.current_load}/{capacity} ({state.utilization:.1%})")
            print(f"  Pending Completions: {state.pending_completions}")
            print(f"  Avg Processing Time: {state.avg_processing_time:.2f}s")
            print(f"  Recent Requests: {state.recent_requests}")
        
        # Event statistics
        total_events = sum(self.event_counts.values())
        if total_events > 0:
            print(f"\nEvent Statistics:")
            print(f"  Added: {self.event_counts['request_added']}")
            print(f"  Completed: {self.event_counts['request_completed']}")
            print(f"  Failed: {self.event_counts['request_failed']}")
            
            # Calculate rates
            success_rate = self.event_counts['request_completed'] / max(1, self.event_counts['request_added'])
            failure_rate = self.event_counts['request_failed'] / max(1, total_events)
            
            print(f"  Success Rate: {success_rate:.1%}")
            print(f"  Failure Rate: {failure_rate:.1%}")
        
        print("=" * 60)
    
    def get_queue_summary(self):
        """Get summary of queue states over time"""
        if not self.queue_events:
            return {}
        
        summary = {}
        
        for server_id in range(len(Config.SERVER_CAPACITIES)):
            server_events = [e for e in self.queue_events if e.server_id == server_id]
            
            if server_events:
                valid_events = [e for e in server_events if e.event_type in ['request_added', 'request_completed'] and e.utilization_after >= 0]
                
                if valid_events:
                    utilizations = [e.utilization_after for e in valid_events]
                    queue_lengths = [e.queue_length_after for e in valid_events if e.queue_length_after >= 0]
                    
                    summary[f"server_{server_id}"] = {
                        'avg_utilization': np.mean(utilizations) if utilizations else 0,
                        'max_utilization': np.max(utilizations) if utilizations else 0,
                        'avg_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
                        'max_queue_length': np.max(queue_lengths) if queue_lengths else 0,
                        'total_events': len(server_events),
                        'time_at_capacity': sum(1 for e in valid_events if e.queue_length_after >= Config.SERVER_CAPACITIES[server_id]) / len(valid_events) if valid_events else 0
                    }
        
        return summary
    
    def reset(self):
        """Reset all monitoring data"""
        self.queue_events.clear()
        self.event_counts = {
            'request_added': 0,
            'request_completed': 0,
            'request_failed': 0
        }
        
        # Reset server states
        for server_id in self.server_states:
            state = self.server_states[server_id]
            state.current_load = 0
            state.utilization = 0.0
            state.pending_completions = 0
            state.avg_processing_time = 0.0
            state.recent_requests = 0
            state.active_request_ids = []