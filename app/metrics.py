from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone

class StatsTracker:
    def __init__(self, max_recent=200):
        self.questions_answered = 0
        self.total_chunks = 0
        self._latencies_ms = deque(maxlen=max_recent)
        self._recent_questions = deque(maxlen=max_recent)
        self._by_day = defaultdict(int)

    def record_upload(self, chunks_added:int):
        self.total_chunks += int(chunks_added)

    def record_question(self, question:str, latency_ms:float, top_k:int):
        self.questions_answered += 1
        self._latencies_ms.append(float(latency_ms))
        now = datetime.now(timezone.utc)
        self._recent_questions.append({
            "timestamp": now.isoformat(),
            "question": question,
            "latency_ms": round(latency_ms, 2),
            "top_k": top_k
        })
        self._by_day[now.date()] += 1

    def get_stats(self):
        avg_ms = round(sum(self._latencies_ms)/len(self._latencies_ms), 2) if self._latencies_ms else 0.0
        today = datetime.now(timezone.utc).date()
        last7 = []
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            last7.append({"date": d.isoformat(), "questions": int(self._by_day.get(d, 0))})
        return {
            "total_chunks": int(self.total_chunks),
            "questions_answered": int(self.questions_answered),
            "avg_ms": avg_ms,
            "last7": last7,
            "lastN_questions": list(self._recent_questions)[-20:],
        }

tracker = StatsTracker()

