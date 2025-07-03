import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

export default function PerformanceTracker() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetch("/api/prediction-log")
      .then((res) => res.json())
      .then((data) => {
        const formatted = data.map((entry) => ({
          time: new Date(entry.timestamp).toLocaleTimeString(),
          score: entry.executive_function_score,
        }));
        setHistory(formatted);
      });
  }, []);

  const chartData = {
    labels: history.map((h) => h.time),
    datasets: [
      {
        label: "Executive Function Score",
        data: history.map((h) => h.score),
        borderWidth: 2,
        tension: 0.3,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: true },
      tooltip: { mode: "index", intersect: false },
    },
    scales: {
      y: {
        min: 0,
        max: 1,
      },
    },
  };

  return (
    <div className="p-4 space-y-4">
      <Card>
        <CardContent className="p-4">
          <h2 className="text-xl font-bold mb-2">Cognitive Performance Over Time</h2>
          <Line data={chartData} options={chartOptions} />
        </CardContent>
      </Card>
    </div>
  );
}

