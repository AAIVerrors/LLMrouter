import numpy as np
import wandb
from config import Config

class TrainingPlotter:
    def __init__(self, wandb_available=False):
        self.wandb_available = wandb_available
    
    def plot_training_progress(self, episode_rewards, episode_stats):
        """Plot training progress"""
        if len(episode_rewards) < 5:
            print("  Not enough episodes for plotting yet...")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Episode Rewards over time
            episodes = range(len(episode_rewards))
            axes[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.7, linewidth=1)
            axes[0, 0].set_title('Cumulative Reward per Episode', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add trend line
            if len(episode_rewards) > 10:
                z = np.polyfit(episodes, episode_rewards, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=2, label='Trend')
                axes[0, 0].legend()
            
            # 2. Moving average of rewards
            window = min(20, max(5, len(episode_rewards) // 10))
            if window > 1 and len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                moving_episodes = range(window-1, len(episode_rewards))
                axes[0, 1].plot(moving_episodes, moving_avg, 'g-', linewidth=2)
                axes[0, 1].set_title(f'Moving Average Reward (window={window})', fontsize=14, fontweight='bold')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Moving Avg Reward')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'Not enough data\nfor moving average', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Moving Average Reward', fontsize=14)
            
            # 3. Action distribution over recent episodes
            if len(episode_stats) > 0:
                recent_stats = episode_stats[-min(20, len(episode_stats)):]
                action_dist = np.mean([stat['action_distribution'] for stat in recent_stats], axis=0)
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(action_dist)]
                bars = axes[1, 0].bar(range(len(action_dist)), action_dist, color=colors)
                axes[1, 0].set_title('Recent Action Distribution', fontsize=14, fontweight='bold')
                axes[1, 0].set_xlabel('Server ID')
                axes[1, 0].set_ylabel('Avg Actions per Episode')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
            else:
                axes[1, 0].text(0.5, 0.5, 'No action data\navailable yet', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Action Distribution', fontsize=14)
            
            # 4. Invalid action rate and completed requests over time
            if len(episode_stats) > 0:
                invalid_rates = [stat['invalid_actions'] / Config.EPISODE_LENGTH for stat in episode_stats]
                completed_requests = [stat['completed_requests'] for stat in episode_stats]
                
                ax4_1 = axes[1, 1]
                ax4_2 = ax4_1.twinx()
                
                # Invalid action rate
                line1 = ax4_1.plot(range(len(invalid_rates)), invalid_rates, 'r-', linewidth=2, label='Invalid Action Rate')
                ax4_1.set_ylabel('Invalid Action Rate', color='r')
                ax4_1.tick_params(axis='y', labelcolor='r')
                ax4_1.grid(True, alpha=0.3)
                
                # Completed requests
                line2 = ax4_2.plot(range(len(completed_requests)), completed_requests, 'b-', linewidth=2, label='Completed Requests')
                ax4_2.set_ylabel('Completed Requests', color='b')
                ax4_2.tick_params(axis='y', labelcolor='b')
                
                ax4_1.set_xlabel('Episode')
                ax4_1.set_title('Training Metrics', fontsize=14, fontweight='bold')
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax4_1.legend(lines, labels, loc='upper left')
            else:
                axes[1, 1].text(0.5, 0.5, 'No metrics data\navailable yet', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Training Metrics', fontsize=14)
            
            plt.tight_layout()
            plt.savefig('training_progress.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Training progress plot saved to training_progress.png")
            
            # Log to wandb if available
            if self.wandb_available:
                try:
                    wandb.log({"training_progress": wandb.Image('training_progress.png')})
                    print(f"  Plot uploaded to wandb")
                except Exception as e:
                    print(f"  Failed to upload plot to wandb: {e}")
                    
        except ImportError:
            print(f"  Matplotlib not available - cannot create plots")
        except Exception as e:
            print(f"  Plotting failed: {e}")
    
    def plot_final_analysis(self, episode_rewards, episode_stats):
        """Create comprehensive final analysis plots"""
        if len(episode_rewards) < 10:
            print("Not enough data for final analysis")
            return
            
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Reward progression with trend
            episodes = range(len(episode_rewards))
            axes[0, 0].plot(episodes, episode_rewards, 'b-', alpha=0.6, label='Episode Rewards')
            
            # Add moving average
            window = min(20, len(episode_rewards) // 10)
            if window > 1:
                moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
                moving_episodes = range(window-1, len(episode_rewards))
                axes[0, 0].plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            
            axes[0, 0].set_title('Training Progress')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Learning curve analysis
            if len(episode_rewards) >= 20:
                first_quarter = episode_rewards[:len(episode_rewards)//4]
                last_quarter = episode_rewards[-len(episode_rewards)//4:]
                
                axes[0, 1].hist([first_quarter, last_quarter], bins=20, alpha=0.7, 
                               label=['First Quarter', 'Last Quarter'])
                axes[0, 1].set_title('Reward Distribution Comparison')
                axes[0, 1].set_xlabel('Reward')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
            
            # 3. Action distribution evolution
            if episode_stats:
                early_stats = episode_stats[:len(episode_stats)//3] if len(episode_stats) >= 3 else episode_stats[:1]
                late_stats = episode_stats[-len(episode_stats)//3:] if len(episode_stats) >= 3 else episode_stats[-1:]
                
                early_dist = np.mean([stat['action_distribution'] for stat in early_stats], axis=0)
                late_dist = np.mean([stat['action_distribution'] for stat in late_stats], axis=0)
                
                x = np.arange(len(early_dist))
                width = 0.35
                
                axes[0, 2].bar(x - width/2, early_dist, width, label='Early Training', alpha=0.8)
                axes[0, 2].bar(x + width/2, late_dist, width, label='Late Training', alpha=0.8)
                axes[0, 2].set_title('Action Distribution Evolution')
                axes[0, 2].set_xlabel('Server ID')
                axes[0, 2].set_ylabel('Actions per Episode')
                axes[0, 2].legend()
            
            # 4. Performance metrics over time
            if episode_stats:
                invalid_rates = [stat['invalid_actions'] / Config.EPISODE_LENGTH for stat in episode_stats]
                completed_rates = [stat['completed_requests'] for stat in episode_stats]
                
                axes[1, 0].plot(invalid_rates, 'r-', label='Invalid Action Rate')
                axes[1, 0].set_ylabel('Invalid Action Rate', color='r')
                axes[1, 0].tick_params(axis='y', labelcolor='r')
                
                ax_twin = axes[1, 0].twinx()
                ax_twin.plot(completed_rates, 'g-', label='Completed Requests')
                ax_twin.set_ylabel('Completed Requests', color='g')
                ax_twin.tick_params(axis='y', labelcolor='g')
                
                axes[1, 0].set_title('Training Metrics')
                axes[1, 0].set_xlabel('Episode')
            
            # 5. Quality and latency trends
            if episode_stats:
                quality_scores = [np.mean(stat['quality_scores']) for stat in episode_stats]
                avg_latencies = [np.mean([l for l in stat['latencies'] if l > 0]) 
                               for stat in episode_stats if any(l > 0 for l in stat['latencies'])]
                
                axes[1, 1].plot(quality_scores, 'b-', label='Quality Score')
                axes[1, 1].set_ylabel('Quality Score', color='b')
                axes[1, 1].tick_params(axis='y', labelcolor='b')
                
                if avg_latencies:
                    ax_twin2 = axes[1, 1].twinx()
                    ax_twin2.plot(range(len(avg_latencies)), avg_latencies, 'orange', label='Avg Latency')
                    ax_twin2.set_ylabel('Latency (s)', color='orange')
                    ax_twin2.tick_params(axis='y', labelcolor='orange')
                
                axes[1, 1].set_title('Quality vs Latency')
                axes[1, 1].set_xlabel('Episode')
            
            # 6. Final statistics summary
            axes[1, 2].axis('off')
            
            # Calculate summary statistics
            total_episodes = len(episode_rewards)
            avg_reward = np.mean(episode_rewards)
            best_reward = np.max(episode_rewards)
            final_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]
            
            if episode_stats:
                total_invalid = sum(stat['invalid_actions'] for stat in episode_stats)
                total_actions = total_episodes * Config.EPISODE_LENGTH
                invalid_rate = total_invalid / total_actions if total_actions > 0 else 0
                total_completed = sum(stat['completed_requests'] for stat in episode_stats)
            else:
                invalid_rate = 0
                total_completed = 0
            
            summary_text = f"""
                Training Summary:

                Total Episodes: {total_episodes}
                Average Reward: {avg_reward:.3f}
                Best Reward: {best_reward:.3f}
                Final 10-ep Avg: {final_reward:.3f}

                Invalid Action Rate: {invalid_rate:.3f}
                Total Completed Requests: {total_completed}

                Improvement: {final_reward - avg_reward:.3f}
                """
            
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=12, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig('final_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print("Final analysis plot saved to final_analysis.png")
            
            if self.wandb_available:
                try:
                    wandb.log({"final_analysis": wandb.Image('final_analysis.png')})
                    print("Final analysis uploaded to wandb")
                except Exception as e:
                    print(f"Failed to upload final analysis to wandb: {e}")
                    
        except ImportError:
            print("Matplotlib not available - cannot create final analysis")
        except Exception as e:
            print(f"Final analysis plotting failed: {e}")