# backend/analysis.py

from datetime import datetime, timedelta

def correlate_habits_with_scores(habits, scores, time_window_min=5):
    """
    Correlate logged habits with executive function scores.
    - habits: list of dicts with keys: ['habit', 'timestamp']
    - scores: list of dicts with keys: ['executive_function_score', 'timestamp']
    - time_window_min: int, minutes before/after a habit to look for scores

    Returns: dict of average score per habit type
    """
    time_window_ms = time_window_min * 60 * 1000
    habit_map = {}

    for habit_entry in habits:
        habit_time = datetime.fromisoformat(habit_entry['timestamp'])
        habit_label = habit_entry['habit']

        matched_scores = [
            s['executive_function_score']
            for s in scores
            if abs((datetime.fromisoformat(s['timestamp']) - habit_time).total_seconds()) <= time_window_min * 60
        ]

        if matched_scores:
            if habit_label not in habit_map:
                habit_map[habit_label] = []
            habit_map[habit_label].extend(matched_scores)

    # Compute average
    avg_map = {
        habit: round(sum(vals) / len(vals), 4)
        for habit, vals in habit_map.items() if vals
    }

    return avg_map

