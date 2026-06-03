import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class CostTracker:
    """
    Tracks API usage and calculates costs for Groq API calls.
    Stores data in JSON file for persistence across sessions.
    """
    
    # Groq pricing (approximate - based on free tier estimates)
    GROQ_LLM_INPUT_COST = 0.000000015  # per token
    GROQ_LLM_OUTPUT_COST = 0.000000060  # per token
    GROQ_WHISPER_COST_PER_MIN = 0.00167  # ~$0.10 per 60 minutes
    
    def __init__(self, storage_dir: str = "app/storage"):
        """Initialize cost tracker with storage directory."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.costs_file = self.storage_dir / "costs.json"
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load cost data from JSON file or initialize new."""
        if self.costs_file.exists():
            with open(self.costs_file, "r") as f:
                return json.load(f)
        return {
            "sessions": [],
            "total_cost": 0.0,
            "total_tokens": {"input": 0, "output": 0},
            "task_breakdown": {},
        }
    
    def _save_data(self):
        """Save cost data to JSON file."""
        with open(self.costs_file, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def log_llm_call(
        self,
        task: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_sec: float = 0.0,
    ):
        """Log an LLM API call with token counts."""
        input_cost = input_tokens * self.GROQ_LLM_INPUT_COST
        output_cost = output_tokens * self.GROQ_LLM_OUTPUT_COST
        total_cost = input_cost + output_cost
        
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "type": "llm",
            "task": task,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_sec": duration_sec,
            "input_cost": round(input_cost, 8),
            "output_cost": round(output_cost, 8),
            "total_cost": round(total_cost, 8),
        }
        
        self.data["sessions"].append(call_record)
        self.data["total_cost"] += total_cost
        self.data["total_tokens"]["input"] += input_tokens
        self.data["total_tokens"]["output"] += output_tokens
        
        # Update task breakdown
        if task not in self.data["task_breakdown"]:
            self.data["task_breakdown"][task] = {
                "count": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
            }
        self.data["task_breakdown"][task]["count"] += 1
        self.data["task_breakdown"][task]["total_cost"] += total_cost
        self.data["task_breakdown"][task]["total_tokens"] += (
            input_tokens + output_tokens
        )
        
        self._save_data()
    
    def log_audio_call(self, duration_sec: float):
        """Log a Whisper audio transcription call."""
        cost = (duration_sec / 60) * self.GROQ_WHISPER_COST_PER_MIN
        
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "type": "audio",
            "task": "audio_transcription",
            "model": "whisper-large-v3",
            "duration_sec": duration_sec,
            "total_cost": round(cost, 8),
        }
        
        self.data["sessions"].append(call_record)
        self.data["total_cost"] += cost
        
        if "audio_transcription" not in self.data["task_breakdown"]:
            self.data["task_breakdown"]["audio_transcription"] = {
                "count": 0,
                "total_cost": 0.0,
                "duration_sec": 0.0,
            }
        self.data["task_breakdown"]["audio_transcription"]["count"] += 1
        self.data["task_breakdown"]["audio_transcription"]["total_cost"] += cost
        self.data["task_breakdown"]["audio_transcription"]["duration_sec"] += (
            duration_sec
        )
        
        self._save_data()
    
    def get_session_cost(self) -> Dict:
        """Get current session statistics."""
        return {
            "total_cost": round(self.data["total_cost"], 8),
            "total_input_tokens": self.data["total_tokens"]["input"],
            "total_output_tokens": self.data["total_tokens"]["output"],
            "total_tokens": (
                self.data["total_tokens"]["input"]
                + self.data["total_tokens"]["output"]
            ),
            "total_calls": len(self.data["sessions"]),
        }
    
    def get_task_breakdown(self) -> Dict:
        """Get cost breakdown by task type."""
        return self.data["task_breakdown"]
    
    def get_daily_stats(self) -> Dict:
        """Get stats grouped by day."""
        daily = {}
        for call in self.data["sessions"]:
            date = call["timestamp"][:10]  # YYYY-MM-DD
            if date not in daily:
                daily[date] = {"cost": 0.0, "calls": 0, "tokens": 0}
            
            daily[date]["cost"] += call.get("total_cost", 0)
            daily[date]["calls"] += 1
            daily[date]["tokens"] += call.get("input_tokens", 0) + call.get(
                "output_tokens", 0
            )
        
        return daily
    
    def clear_history(self):
        """Clear all cost history."""
        self.data = {
            "sessions": [],
            "total_cost": 0.0,
            "total_tokens": {"input": 0, "output": 0},
            "task_breakdown": {},
        }
        self._save_data()


# Global instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
